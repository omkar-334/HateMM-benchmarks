import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalSentimentModel(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=768, n_heads=12, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.linguistic_projection = nn.Linear(feature_dim, hidden_dim).to(device)
        self.acoustic_bilstm = nn.LSTM(feature_dim, hidden_dim // 2, bidirectional=True, batch_first=True).to(device)
        self.visual_bilstm = nn.LSTM(feature_dim, hidden_dim // 2, bidirectional=True, batch_first=True).to(device)

        self.uff_projection = nn.Linear(hidden_dim * 3, hidden_dim).to(device)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4).to(device)

        self.multimodal_classifier = TaskSpecificLayer(hidden_dim, device)
        self.linguistic_classifier = TaskSpecificLayer(hidden_dim, device)
        self.acoustic_classifier = TaskSpecificLayer(hidden_dim, device)
        self.visual_classifier = TaskSpecificLayer(hidden_dim, device)

        self.register_buffer("pos_center", torch.zeros(feature_dim, device=device))
        self.register_buffer("neg_center", torch.zeros(feature_dim, device=device))

        self.to(device)

    def forward(self, linguistic, acoustic, visual):
        linguistic = linguistic.float().to(self.device)
        acoustic = acoustic.float().to(self.device)
        visual = visual.float().to(self.device)

        if linguistic.dim() == 2:
            linguistic = linguistic.unsqueeze(1)
        if acoustic.dim() == 2:
            acoustic = acoustic.unsqueeze(1)
        if visual.dim() == 2:
            visual = visual.unsqueeze(1)

        l_features = self.linguistic_projection(linguistic)
        a_features, _ = self.acoustic_bilstm(acoustic)
        v_features, _ = self.visual_bilstm(visual)

        multimodal_features = self.unified_feature_fusion(l_features, a_features, v_features)

        l_guided = self.transformer_encoder(l_features.permute(1, 0, 2))
        a_guided = self.guided_attention(l_features, a_features)
        v_guided = self.guided_attention(l_features, v_features)

        multimodal_out = self.multimodal_classifier(multimodal_features)
        linguistic_out = self.linguistic_classifier(l_guided.permute(1, 0, 2))
        acoustic_out = self.acoustic_classifier(a_guided)
        visual_out = self.visual_classifier(v_guided)

        if self.training:
            return multimodal_out, linguistic_out, acoustic_out, visual_out
        return multimodal_out

    def unified_feature_fusion(self, l, a, v):
        fused = torch.cat([l, a, v], dim=-1)

        fused = self.uff_projection(fused)

        return fused

    def guided_attention(self, query, key_value):
        attention_weights = F.softmax(torch.matmul(query, key_value.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, device=self.device).float()), dim=-1)
        return torch.matmul(attention_weights, key_value)


class TaskSpecificLayer(nn.Module):
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, 1).to(device)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x)


class SLGM:
    def __init__(self, model):
        self.model = model

    def update_centers(self, features, labels):
        features = features.float().to(self.model.device)
        labels = labels.float().to(self.model.device)

        pos_mask = labels > 0
        neg_mask = labels < 0

        if pos_mask.any():
            self.model.pos_center = features[pos_mask].mean(0)
        if neg_mask.any():
            self.model.neg_center = features[neg_mask].mean(0)

    def generate_unimodal_labels(self, features, multimodal_labels):
        features = features.float().to(self.model.device)
        multimodal_labels = multimodal_labels.float().to(self.model.device)

        sp = torch.sum(torch.sqrt(features * self.model.pos_center), dim=1)
        sn = torch.sum(torch.sqrt(features * self.model.neg_center), dim=1)

        sm = torch.where(sp > sn, sp, sn)
        unimodal_labels = (multimodal_labels + (sp - sm)) * (2 * multimodal_labels + sm) / sm

        return unimodal_labels
