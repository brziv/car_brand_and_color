import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import train_dataloader, val_dataloader, test_dataloader
from train import run_epoch
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns

# config
num_classes = 22
num_epochs = 50
lr = 1e-4
# resnet50, regnet, resnext50, efficientnet
model_name = "efficientnet"

# early stopping
patience = 2
best_val_loss = float("inf")
no_improve = 0

# model
model = get_model(model_name, num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# create dicts
os.makedirs("results", exist_ok=True)
train_metrics = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []}
val_metrics = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []}

# main loop
for epoch in range(num_epochs):
    train_loss, train_acc, train_prec, train_rec, train_f1, _ = run_epoch(
        model, train_dataloader, criterion, optimizer, mode="Train"
    )
    val_loss, val_acc, val_prec, val_rec, val_f1, _ = run_epoch(
        model, val_dataloader, criterion, optimizer=None, mode="Validation"
    )

    # save metrics to dict
    train_metrics["loss"].append(train_loss)
    train_metrics["acc"].append(train_acc)
    train_metrics["precision"].append(train_prec)
    train_metrics["recall"].append(train_rec)
    train_metrics["f1"].append(train_f1)

    val_metrics["loss"].append(val_loss)
    val_metrics["acc"].append(val_acc)
    val_metrics["precision"].append(val_prec)
    val_metrics["recall"].append(val_rec)
    val_metrics["f1"].append(val_f1)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Precision: {train_prec:.2f}%, Recall: {train_rec:.2f}%, F1: {train_f1:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Precision: {val_prec:.2f}%, Recall: {val_rec:.2f}%, F1: {val_f1:.2f}%\n")

    # early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        # save best model
        torch.save(model.state_dict(), f"results/{model_name}_best.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# test
test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = run_epoch(
    model, test_dataloader, criterion, optimizer=None, mode="Test"
)
print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Precision: {test_prec:.2f}%, Recall: {test_rec:.2f}%, F1: {test_f1:.2f}%")

# save test metrics
test_metrics = {
    "loss": test_loss,
    "acc": test_acc,
    "precision": test_prec,
    "recall": test_rec,
    "f1": test_f1,
    "confusion_matrix": test_cm.tolist()
}
with open(f"results/{model_name}_test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

# plot metrics
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
metrics_to_plot = ["loss", "acc", "precision", "recall", "f1"]

for i, metric in enumerate(metrics_to_plot):
    axes[i].plot(train_metrics[metric], label="Train")
    axes[i].plot(val_metrics[metric], label="Validation")
    axes[i].set_title(f"{metric.capitalize()} per Epoch")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel(metric.capitalize())
    axes[i].legend()
    axes[i].grid(True)

# remove unused subplot
axes[-1].axis("off")

plt.tight_layout()
plt.savefig(f"results/{model_name}_metrics.png")
plt.close()

# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"results/{model_name}_cmatrix.png")
plt.close()
