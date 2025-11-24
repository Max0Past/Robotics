import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from lib.feature_extractor import ResNetFeatureExtractor
from lib.dataset import CIFAR10

# 0. Load Data
print("Loading CIFAR-10 dataset...")
cifar10 = CIFAR10()
images_train, labels_train, images_val, labels_val, images_test, labels_test = cifar10.load_dataset()

# 1. Dataset Class
class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to torch tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 2. Data Preparation
# Standard CIFAR-10 normalization
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(*stats, inplace=True)
])

val_transform = transforms.Compose([
    transforms.Normalize(*stats)
])

# Create Datasets
train_dataset = CIFAR10Dataset(images_train, labels_train, transform=train_transform)
val_dataset = CIFAR10Dataset(images_val, labels_val, transform=val_transform)
test_dataset = CIFAR10Dataset(images_test, labels_test, transform=val_transform)

# Create DataLoaders
BATCH_SIZE = 128
# num_workers=0 for Windows compatibility
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)

# 3. Model Setup
print("Initializing ResNet-18 model...")
feature_extractor = ResNetFeatureExtractor()
model = feature_extractor.model
device = feature_extractor.device

# 4. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
epochs = 30
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=steps_per_epoch)

# 5. Training Loop
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

print(f"Training on {device}...")
best_val_acc = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': train_loss/len(train_loader), 'acc': train_correct/train_total})
        
    val_acc = evaluate(val_loader, model)
    print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # torch.save(model.state_dict(), 'best_resnet_cifar10.pth')

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# 6. Final Test
test_acc = evaluate(test_loader, model)
print(f"Final Test Accuracy: {test_acc:.4f}")
