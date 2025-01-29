import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold


class DataLoader:
    def __init__(self, config):
        self.config = config

        self.transcriptions_path = "transcriptions.csv"
        self.bert_features_path = "bert_features.jsonl"
        self.audio_features_path = "audio_features.p"
        self.video_features_path = "features"
        self.indices_file = "data/split_indices.p"

        self.dataset_dict = self.load_transcriptions()

        self.bert_embeddings = self.load_bert_features()
        self.audio_features = self.load_audio_features()

        self.data_inputs: List[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = []
        self.labels: List[int] = []
        self.ids: List[str] = []

        self.process_data()

        self.split_indices = self.create_or_load_splits()

    def load_transcriptions(self) -> Dict[str, str]:
        """Load transcriptions from CSV and create dataset dictionary."""
        df = pd.read_csv(self.transcriptions_path)
        dataset_dict = {}

        for _, row in df.iterrows():
            file_id = row["file_name"].removesuffix(".wav")
            label = 1 if file_id.startswith("hate") else 0
            dataset_dict[file_id] = label
        return dataset_dict

    def load_bert_features(self) -> Dict[str, np.ndarray]:
        """Load BERT embeddings from jsonlines file."""
        bert_features = {}
        with jsonlines.open(self.bert_features_path) as reader:
            for item in reader:
                features = item["features"][0]  # CLS token
                embedding = np.mean([np.array(features["layers"][layer]["values"]) for layer in range(4)], axis=0)
                bert_features[item["linex_index"]] = embedding
        return bert_features

    def load_audio_features(self) -> Dict[str, np.ndarray]:
        """Load audio features from pickle file."""
        with open(self.audio_features_path, "rb") as file:
            return pickle.load(file)

    def load_video_feature(self, id_: str) -> Optional[np.ndarray]:
        """Load individual video feature file."""
        feature_path = os.path.join(self.video_features_path, f"{id_}_resnet_pool5.pt")
        if os.path.exists(feature_path):
            try:
                feature = torch.load(feature_path)
                return feature.cpu().numpy()
            except Exception as e:
                print(f"Error loading video feature for ID {id_}: {e}")
                return None
        return None

    def process_data(self):
        """Process all data and store in memory."""
        for id_, data in self.dataset_dict.items():
            self.ids.append(id_)

            bert_id = int(id_.split("_")[-1])
            bert_feature = self.bert_embeddings.get(bert_id)
            if bert_feature is None:
                print("bert missing", end="---")

            audio_feature = self.audio_features.get(id_)
            if audio_feature is None:
                print("audio missing", end="---")

            video_feature = self.load_video_feature(id_)
            if video_feature is None:
                print("video missing", end="---")
            else:
                video_feature = np.mean(video_feature, axis=0)

            self.data_inputs.append((bert_feature, audio_feature, video_feature))
            self.labels.append(int(data))

    def create_or_load_splits(self, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """Create or load existing train/test splits."""
        if os.path.exists(self.indices_file):
            with open(self.indices_file, "rb") as file:
                return pickle.load(file)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_indices = list(skf.split(self.data_inputs, self.labels))

        os.makedirs(os.path.dirname(self.indices_file), exist_ok=True)
        with open(self.indices_file, "wb") as file:
            pickle.dump(split_indices, file, protocol=2)

        return split_indices

    def get_fold(self, fold_idx: int) -> Tuple[Tuple[List[Any], List[int]], Tuple[List[Any], List[int]]]:
        """Get train/test data for a specific fold."""
        train_idx, test_idx = self.split_indices[fold_idx]

        train_inputs = [self.data_inputs[i] for i in train_idx]
        train_labels = [self.labels[i] for i in train_idx]

        test_inputs = [self.data_inputs[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]
        return (train_inputs, train_labels), (test_inputs, test_labels)

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return the dimensions of each feature type."""
        sample_input = self.data_inputs[0]
        dims = {
            "bert": sample_input[0].shape[0] if sample_input[0] is not None else 0,
            "audio": sample_input[1].shape[0] if sample_input[1] is not None else 0,
            "video": sample_input[2].shape[0] if sample_input[2] is not None else 0,
        }
        return dims

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        return {"hate": sum(1 for label in self.labels if label == 1), "non_hate": sum(1 for label in self.labels if label == 0), "total": len(self.labels)}
