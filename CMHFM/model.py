from typing import Literal

import torch
import torch.nn as nn
from dataset import clear_gpu_memory
from transformers import BertModel, BertTokenizer

EMBED_DIM_TEXT = 768
EMBED_DIM_AUDIO = 195
EMBED_DIM_VIDEO = 2304
HIDDEN_DIM = 128
HIDDEN_DIMS = [1024, 768]
OUTPUT_DIM = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

HIDDEN_DIM = 128  # Input dim for all modalities is 128
CLASS_OUTPUT_DIM = 1  # Binary classification output
NUM_HEADS = 4


"""The CMHFM contains four modules, i.e., Unimodal Feature Learning
 (UFL), Inter-Modal Interaction (IMI), Multi-Head Attention (MHA), and Multi-Tasking Assisted Learning (HTAL). We model the depth
 feature of the unimodal in Section 3.1. Then, we use TFN and multi-head attention to explore cross-modal interactions in Sections 3.2
 and 3.3. Finally, we introduce multi-task learning and cross-modal hierarchical fusion to aid the final sentiment prediction in Section
 3.4.
"""


"""Since pre-trained BERT has a powerful feature representation, we use it to extract the initial features of the text 𝑋𝑡 ∈ 𝑅𝑙𝑡× 𝑑𝑡,
 where 𝑙𝑡 and 𝑑𝑡 denote the sequence length and embedding dimension of the text feature, respectively. Bidirectional Gate Recurrent
 Unit (BiGRU) can effectively capture contextual information relevance. Therefore, we introduce BiGRU to further mine the depth
 features of the text 𝐷𝑡, which is calculated as below:
 𝐷𝑡 = BiGRU(𝑋𝑡;𝜃BiGRU)
 where 𝜃BiGRU is the parameter of the BiGRU model."""


# BERT and  Bidirectional GRU
class BiGRUModel(nn.Module):
    def __init__(self):
        super(BiGRUModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)
        self.bert = self.bert.to(DEVICE)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.bigru = nn.GRU(
            input_size=EMBED_DIM_TEXT,
            hidden_size=OUTPUT_DIM // 2,
            bidirectional=True,
            batch_first=True,
        ).to(DEVICE)
        self.max_length = HIDDEN_DIM

    def process_in_batches(self, text_list, batch_size=BATCH_SIZE):
        """Process text in batches to avoid OOM"""
        all_outputs = []

        all_encoded = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        for i in range(0, len(text_list), batch_size):
            batch_encoded = {"input_ids": all_encoded["input_ids"][i : i + batch_size], "attention_mask": all_encoded["attention_mask"][i : i + batch_size]}

            batch_encoded = {k: v.to(DEVICE) for k, v in batch_encoded.items()}

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # BERT output
                    bert_output = self.bert(**batch_encoded)["last_hidden_state"]

                    # GRU output
                    bigru_output, _ = self.bigru(bert_output)

                    last_hidden = bigru_output[:, -1, :]
                    all_outputs.append(last_hidden.cpu())

            clear_gpu_memory()

            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{len(text_list)} texts...")

        return torch.cat(all_outputs, dim=0)

    def forward(self, text_list):
        return self.process_in_batches(text_list)


""" For audio modality, following the previous works (Wu et al., 2022), we use the LibROSA toolkit to mine features such as Mel
 Frequency Cepstral Coefficients (MFCCs), energy, and pitch. The initial audio features are described by 𝑋𝑎 ∈ 𝑅𝑙𝑎× 𝑑𝑎, where 𝑙𝑎 and
 𝑑𝑎 are the sequence length and embedding dimension of the audio feature, respectively. For vision modality, FFmpeg toolkit is used
 to extract frames from the video clips, and MTCNN is applied for face recognition and alignment. The 2D keypoint coordinates (in
 pixels), 3D keypoint coordinates (in millimeters), head pose (position and rotation), facial action units, and eye gaze are obtained
 by the OpenFace toolkit. Let 𝑋𝑣 ∈ 𝑅𝑙𝑣× 𝑑𝑣 denote the original video features, where 𝑙𝑣 and 𝑑𝑣 are the sequence length and embedding
 dimension of the video feature, respectively. The background noise in the audio modality and the presence of undetected faces in the
 aligned images are interference information that can adversely affect the accuracy of classification. Deep neural networks (DNN) can
 tolerate the presence of noisy information to a certain extent. Therefore, we utilize a three-layer DNN to extract the depth features
 of the audio and video modality, respectively. The equations are shown as follows:
 𝐷𝑎 =DNN(𝑋𝑎;𝜃𝑎 )
 𝐷𝑣 =DNN(𝑋𝑣;𝜃𝑣 )
 where 𝜃𝑎 and 𝜃𝑣 are parameters of the DNN network."""


class DNN(nn.Module):
    def __init__(self, type: Literal["audio", "video"]):
        super(DNN, self).__init__()
        input_dim = EMBED_DIM_AUDIO if type == "audio" else EMBED_DIM_VIDEO

        self.dnn = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[1], OUTPUT_DIM),
        ).to(DEVICE)

    def process_in_batches(self, x, batch_size=BATCH_SIZE):
        """Process features in batches"""
        all_outputs = []
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(DEVICE)
            if batch.dim() == 4:  # Video input has 4 dimensions (batch_size, 48, 48, 1), Only perform for video
                batch = batch.view(batch.size(0), -1)  # Flatten each image to a 1D vector of length 2304

            print(batch.shape)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = self.dnn(batch)
                    all_outputs.append(output.cpu())
            clear_gpu_memory()
        return torch.cat(all_outputs, dim=0)

    def forward(self, x):
        return self.process_in_batches(x)


"""3.2. Inter-modal Interaction

After the BERT model and DNN network, we obtain the depth feature representation of text, audio, and vision as Dt, Da, and Dv, respectively. Cross-modal fusion learns interaction information between text, audio, and vision modalities. We introduce TFN to detect inter-modal interactions. TFN is not only effective in capturing useful information between modalities but also simple to operate.

We perform a Cartesian product operation on the feature matrices of two modalities. The text-audio feature vectors, audio-vision feature vectors, and vision-text feature vectors are expressed as follows:

Dv_t = Dv ⊗ Dt, where Dv_t ∈ R^dvt
Dt_a = Dt ⊗ Da, where Dt_a ∈ R^dta
Da_v = Da ⊗ Dv, where Da_v ∈ R^dav"""


class TensorFusionNetwork(nn.Module):
    def __init__(self):
        super(TensorFusionNetwork, self).__init__()
        self.fc = nn.Linear(HIDDEN_DIM * HIDDEN_DIM, HIDDEN_DIM).to(DEVICE)

    def forward(self, modality1, modality2):
        modality1, modality2 = modality1.float(), modality2.float()
        fused = torch.einsum("bi,bj->bij", modality1, modality2)
        fused = fused.view(fused.size(0), -1)
        return self.fc(fused)


""" 3.3. Multi-head attention"""


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).to(DEVICE)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output


"""3.4. Multi-tasking Assisted Learning

TFN dynamically learns the important features between modalities, and multi-head attention captures critical information more comprehensively. Thus, we introduce TFN and multi-head attention to explore inter-modal interaction and bimodal interaction, respectively.

Cross-modal interactions reduce the differences between modalities, but they introduce noise and generate conflicting information that affects the accuracy of sentiment classification. To reduce the impact of negative information, we introduce multi-task learning to improve the robustness of the model.

We execute a cross-modal hierarchical fusion strategy to complement information from different layers while utilizing unimodal, bimodal, and global tasks to assist the final sentiment analysis. The calculations are as follows:

Unimodal fusion F_uni, bimodal fusion F_bi, and global fusion F_mul are calculated as:
F_uni = Concat(Dt, Da, Dv)
F_bi = Concat(Dv_t, Dt_a, Da_v)
F_mul = Concat(Mv_t, Mt_a, Ma_v)

Unimodal prediction y_uni, bimodal prediction y_bi, and global prediction y_mul are expressed as:
y_uni = W1 · F_uni + b1
y_bi = W2 · F_bi + b2
y_mul = W3 · F_mul + b3

Where W1, W2, and W3 are the learnable parameters, and b1, b2, and b3 denote the biases.

The final classification is determined jointly by y_uni, y_bi, and y_mul, which is named y_fusion. It is calculated as:
y_fusion = W4 · F_fusion + b4

Where F_fusion is cascaded from y_uni, y_bi, and y_mul. W4 is the learnable parameter, and b4 denotes the biases.

We introduce three subtasks to aid the final sentiment prediction. Thus, the training loss consists of four parts, which is defined as follows:
L = Loss(F_uni) + Loss(F_bi) + Loss(F_mul) + Loss(F_fusion)"""


class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super(MultiModalSentimentModel, self).__init__()
        self.tfn = TensorFusionNetwork()
        self.attention = MultiHeadAttention(embed_dim=HIDDEN_DIM * 3, num_heads=NUM_HEADS)

        self.fc_unimodal = nn.Linear(HIDDEN_DIM * 3, CLASS_OUTPUT_DIM).to(DEVICE)
        self.fc_bimodal = nn.Linear(HIDDEN_DIM * 3, CLASS_OUTPUT_DIM).to(DEVICE)
        self.fc_global = nn.Linear(HIDDEN_DIM * 3, CLASS_OUTPUT_DIM).to(DEVICE)
        self.fc_fusion = nn.Linear(CLASS_OUTPUT_DIM * 3, CLASS_OUTPUT_DIM).to(DEVICE)

    def forward(self, text_features, audio_features, video_features):
        text_features, audio_features, video_features = text_features.to(DEVICE), audio_features.to(DEVICE), video_features.to(DEVICE)

        # Unimodal Fusion
        f_unimodal = torch.cat([text_features, audio_features, video_features], dim=1)

        # Bimodal Fusion
        f_text_audio = self.tfn(text_features, audio_features)
        f_audio_video = self.tfn(audio_features, video_features)
        f_video_text = self.tfn(video_features, text_features)
        f_bimodal = torch.cat([f_text_audio, f_audio_video, f_video_text], dim=1)

        # Global Fusion with attention
        f_global = torch.cat([f_text_audio, f_audio_video, f_video_text], dim=1).unsqueeze(1)
        f_global = self.attention(f_global, f_global, f_global).squeeze(1)

        # Predictions
        y_unimodal = self.fc_unimodal(f_unimodal.float())
        y_bimodal = self.fc_bimodal(f_bimodal.float())
        y_global = self.fc_global(f_global.float())

        # Final fusion
        f_fusion = torch.cat([y_unimodal, y_bimodal, y_global], dim=1)
        y_fusion = self.fc_fusion(f_fusion)

        return y_unimodal, y_bimodal, y_global, y_fusion
