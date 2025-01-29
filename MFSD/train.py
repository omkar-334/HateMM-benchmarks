import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Config:
    """Simplified configuration class."""

    def __init__(self):
        self.model = "SVM"
        self.runs = 1
        self.num_classes = 2
        self.svm_c = 10.0
        self.svm_scale = True
        self.fold = None

        # Feature flags
        self.use_bert = True
        self.use_target_audio = True
        self.use_target_video = True


class SVMTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.result_file = "output/svm_results.json"

    def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> Any:
        """Train SVM model."""
        pipeline = make_pipeline(StandardScaler() if self.config.svm_scale else None, svm.SVC(C=self.config.svm_c, gamma="scale", kernel="rbf"))
        return pipeline.fit(train_features, train_labels)

    def evaluate(self, model: Any, test_features: np.ndarray, test_labels: np.ndarray) -> Tuple[Dict[str, Any], str]:
        """Evaluate model and return metrics."""
        predictions = model.predict(test_features)

        print("\nConfusion Matrix:")
        print(confusion_matrix(test_labels, predictions))

        report_dict = classification_report(test_labels, predictions, output_dict=True, digits=3)
        report_str = classification_report(test_labels, predictions, digits=3)
        print("\nClassification Report:")
        print(report_str)

        return report_dict, report_str

    def concat_features(self, features: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Concatenate different modality features."""
        bert_emb, audio_feat, video_feat = features
        feature_list = []

        if self.config.use_bert:
            if bert_emb is not None:
                bert_emb = np.asarray(bert_emb, dtype=np.float32).flatten()
            else:
                bert_emb = np.zeros(768, dtype=np.float32)
            feature_list.append(bert_emb)

        if self.config.use_target_audio:
            if audio_feat is not None:
                audio_feat = np.asarray(audio_feat, dtype=np.float32)
                audio_feat = audio_feat.flatten()
            else:
                audio_feat = np.zeros(283 * 11, dtype=np.float32)
            feature_list.append(audio_feat[:3113])

        if self.config.use_target_video:
            if video_feat is not None:
                video_feat = np.asarray(video_feat, dtype=np.float32).flatten()
            else:
                video_feat = np.zeros(2048, dtype=np.float32)
            feature_list.append(video_feat)

        if not feature_list:
            raise ValueError("No features selected for training")

        return np.concatenate(feature_list, axis=0)

    def train_and_evaluate(self, data_loader) -> None:
        """Run training and evaluation across all folds."""
        all_results = []

        for fold_idx in range(len(data_loader.split_indices)):
            self.config.fold = fold_idx + 1
            print(f"\nProcessing Fold {self.config.fold}")

            (train_inputs, train_labels), (test_inputs, test_labels) = data_loader.get_fold(fold_idx)

            train_features = np.array([self.concat_features(x) for x in train_inputs])
            test_features = np.array([self.concat_features(x) for x in test_inputs])

            model = self.train(train_features, train_labels)

            result_dict, _ = self.evaluate(model, test_features, test_labels)
            all_results.append(result_dict)

        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        with open(self.result_file, "w") as f:
            json.dump(all_results, f)

        self.print_final_results(all_results)

    def print_final_results(self, results: List[Dict[str, Any]]) -> None:
        """Print averaged results across all folds."""
        metrics = ["precision", "recall", "f1-score"]
        avg_metrics = {metric: [] for metric in metrics}

        for result in results:
            for metric in metrics:
                avg_metrics[metric].append(result["weighted avg"][metric])

        print("\nFinal Results (averaged across folds):")
        print("=" * 50)
        for metric in metrics:
            mean_value = np.mean(avg_metrics[metric])
            std_value = np.std(avg_metrics[metric])
            print(f"Weighted {metric:10}: {mean_value:.3f} ± {std_value:.3f}")
