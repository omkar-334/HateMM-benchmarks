import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=1e-3, weight_decay=1e-5, patience=10, num_epochs=100, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=patience, factor=0.1, verbose=True)

        self.best_val_loss = float("inf")
        self.best_model_path = "best_model.pt"

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for text, audio, video, labels in pbar:
            text = text.float().to(self.device)
            audio = audio.float().to(self.device)
            video = video.float().to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(text, video, audio)
            loss = self.criterion(outputs.squeeze(), labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def train(self):
        print("Starting training...")
        for epoch in range(self.num_epochs):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("-" * 50)

        self.model.load_state_dict(torch.load(self.best_model_path)["model_state_dict"])
        self.evaluate(self.test_loader, phase="Test")

    def validate(self):
        val_loss, metrics = self.evaluate(self.val_loader, phase="Validation")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"\nSaving best model with validation loss: {val_loss:.4f}")
            torch.save({"model_state_dict": self.model.state_dict(), "val_loss": val_loss, "metrics": metrics}, self.best_model_path)

        return val_loss, metrics

    def evaluate(self, data_loader, phase="Validation"):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for text, audio, video, labels in data_loader:
                text = text.float().to(self.device)
                audio = audio.float().to(self.device)
                video = video.float().to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(text, video, audio)
                loss = self.criterion(outputs.squeeze(), labels.float())

                total_loss += loss.item() * labels.size(0)
                predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()  # Apply sigmoid for binary classification
                labels = labels.cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels)

        avg_loss = total_loss / len(data_loader.dataset)

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        metrics = self.calculate_metrics(all_predictions, all_labels)
        self.print_metrics(metrics, phase)

        return avg_loss, metrics

    @staticmethod
    def calculate_metrics(predictions, labels):
        preds = (predictions > 0.5).astype(int)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def print_metrics(metrics, phase):
        print(f"\n{phase} Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("-" * 50)
