from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class CarDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir
        # self.samples = [(" ".join(p[:-1]), int(p[-1])) for p in (line.strip().split() for line in open(label_file))]
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                label = parts[-1]
                img_name = " ".join(parts[:-1])
                self.samples.append((img_name, int(label)))
        
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_name, label = self.samples[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
# input
train_dataset = CarDataset("images", "annot/train.txt")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = CarDataset("images", "annot/val.txt")
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = CarDataset("images", "annot/test.txt")
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)