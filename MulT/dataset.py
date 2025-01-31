import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


class Multimodal_Dataset(Dataset):
    def __init__(self):
        df = pickle.load(open("features.pkl", "rb"))
        df["hate"] = df["filename"].apply(lambda x: 1 if x.startswith("hate") else 0)

        text_features, audio_features, video_features, labels = [], [], [], []

        for i, row in df.iterrows():
            audio = np.array(row["audio_features"])
            video = np.array(row["video_features"])

            audio_mean = np.mean(audio, axis=0)
            video_mean = np.mean(video, axis=0)

            if (not audio_mean.shape) or (not video_mean.shape):
                continue

            labels.append(row["hate"])
            text_features.append(row["text_features"])
            audio_features.append(audio_mean)
            video_features.append(video_mean)

        self.text = torch.tensor(np.array(text_features), dtype=torch.float32).to(device)
        self.audio = torch.tensor(np.array(audio_features), dtype=torch.float32).to(device)
        self.video = torch.tensor(np.array(video_features), dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

        self.device = device

    def __len__(self):
        return len(self.labels)

    def get_n_modalities(self):
        return 3

    def get_seq_len(self):
        """
        Since mean pooling is applied, there is no sequence length.
        This method now returns a constant sequence length of 1 for compatibility.
        """
        return 1, 1, 1

    def get_dim(self):
        """
        Returns the feature dimensions of each modality.
        Since the features are now pooled, we return the length of the fixed feature vectors.
        """
        text_dim = self.text[0].shape[0]
        audio_dim = self.audio[0].shape[0]
        vision_dim = self.video[0].shape[0]
        return text_dim, audio_dim, vision_dim

    def __getitem__(self, idx):
        """
        Returns the features and label at index `idx` as a tuple.
        The features are returned as a tuple (text, audio, video), and the label is the target.
        """
        return (
            self.text[idx],
            self.audio[idx],
            self.video[idx].squeeze(-1).unsqueeze(0),
            self.labels[idx],
        )
