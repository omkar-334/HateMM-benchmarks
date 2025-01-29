import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

EMBED_DIM_TEXT = 768
EMBED_DIM_AUDIO = 128
EMBED_DIM_VIDEO = 128
HIDDEN_DIM = 256
NUM_HEADS = 8
OUTPUT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, scaler, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        all_labels = []
        all_preds = []

        for text, audio, video, label in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            # Forward pass
            y_unimodal, y_bimodal, y_global, y_fusion = model(text, audio, video)
            label = label.unsqueeze(1).to(DEVICE)

            # Compute loss
            loss = compute_loss(y_unimodal, y_bimodal, y_global, y_fusion, label)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(y_fusion.cpu().detach().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        accuracy, precision, recall, f1 = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return model


def validate(model, dataloader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for text, audio, video, label in dataloader:
            y_unimodal, y_bimodal, y_global, y_fusion = model(text, audio, video)
            label = label.unsqueeze(1).to(DEVICE)

            loss = compute_loss(y_unimodal, y_bimodal, y_global, y_fusion, label)
            total_loss += loss.item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(y_fusion.cpu().detach().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_preds)

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


def compute_loss(y_unimodal, y_bimodal, y_global, y_fusion, labels):
    labels = labels.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(y_unimodal, labels) + loss_fn(y_bimodal, labels) + loss_fn(y_global, labels) + loss_fn(y_fusion, labels)


def compute_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(float)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1
