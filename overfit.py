import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import train_dataloader

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 22
max_steps = 100
lr = 1e-4
model_name = "resnet50"

# train one batch
images, labels = next(iter(train_dataloader))
images, labels = images.to(device), labels.to(device)

model = get_model(model_name, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for step in range(max_steps):
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, preds = outputs.max(1)
    acc = (preds == labels).float().mean().item() * 100

    print(f"Step {step+1}/{max_steps}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
    
    if acc == 100.:
        print("Converged!")
        break
