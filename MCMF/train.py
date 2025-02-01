import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_step(model, optimizer, data, slgm):
    """
    Performs a single training step.
    """
    model.train()
    optimizer.zero_grad()

    data = [d.float().to(model.device) if torch.is_tensor(d) else d for d in data]

    multimodal_out, l_out, a_out, v_out = model(data[0], data[1], data[2])

    label = data[3]

    m_loss = F.mse_loss(multimodal_out, data[3])
    l_loss = F.mse_loss(l_out, label)
    a_loss = F.mse_loss(a_out, label)
    v_loss = F.mse_loss(v_out, label)

    total_loss = m_loss + l_loss + a_loss + v_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def validate(model, valid_loader, slgm):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            linguistic, acoustic, visual, labels = batch

            multimodal_out = model(linguistic.to(model.device), acoustic.to(model.device), visual.to(model.device))

            val_loss += F.mse_loss(multimodal_out, labels.to(model.device)).item()

            all_preds.append(multimodal_out.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_loss /= len(valid_loader)

    print("Validation Metrics:")
    print_metrics_and_loss(all_labels, all_preds, val_loss)

    return val_loss


def train_model(model, optimizer, train_loader, valid_loader, slgm, num_epochs=10):
    """
    Trains the model for a specified number of epochs.
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss = train_step(model, optimizer, batch, slgm)
            train_loss += loss

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = validate(model, valid_loader, slgm)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("-" * 50)


def print_metrics_and_loss(y_true, y_pred, loss=None):
    """
    Prints evaluation metrics and loss.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels.
        y_pred (torch.Tensor or np.ndarray): Predicted labels.
        loss (float, optional): Loss value to print. Defaults to None.
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    y_pred_classes = (y_pred >= 0.5).astype(int)
    y_true_classes = (y_true >= 0.5).astype(int)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if loss is not None:
        print(f"Loss: {loss:.4f}")
