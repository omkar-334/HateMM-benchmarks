import gc
import pickle

import numpy as np
import torch
from model import DNN, BiGRUModel
from torch.utils.data.dataset import Dataset


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Multimodal_Dataset(Dataset):
    def __init__(self):
        text, audio_features, video_features, labels = load_dataset()

        model = BiGRUModel().to(DEVICE)
        model.eval()

        print("text")
        with torch.no_grad():
            self.text = model.process_in_batches(text)
        clear_gpu_memory()

        print("audio")
        self.audio = torch.tensor(audio_features, dtype=torch.float32)
        dnn_audio = DNN("audio").to(DEVICE)
        dnn_audio.eval()
        with torch.no_grad():
            self.audio = dnn_audio.process_in_batches(self.audio)
        clear_gpu_memory()

        print("video")
        self.video = torch.tensor(video_features, dtype=torch.float32)
        dnn_video = DNN("video").to(DEVICE)
        dnn_video.eval()
        with torch.no_grad():
            self.video = dnn_video.process_in_batches(self.video)
        clear_gpu_memory()

        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def get_dim(self):
        text_dim = self.text[0].shape[0]
        audio_dim = self.audio[0].shape[0]
        vision_dim = self.video[0].shape[0]
        return text_dim, audio_dim, vision_dim

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.audio[idx],
            self.video[idx],
            self.labels[idx],
        )


def load_dataset(path="features.pkl"):
    df = pickle.load(open(path, "rb"))
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
        text_features.append(row["transcription"])
        audio_features.append(audio_mean)
        video_features.append(video_mean)

    return text_features, audio_features, video_features, labels
