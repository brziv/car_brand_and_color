from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def run_epoch(model, dataloader, criterion, optimizer=None, mode="Train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    scaler = torch.amp.GradScaler(device, enabled=is_train)
    
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc=mode, leave=False):
            images, labels = images.to(device), labels.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            with torch.amp.autocast(device, enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            _, preds = outputs.max(1)
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()
            total_samples += images.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples * 100
    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, conf_matrix
