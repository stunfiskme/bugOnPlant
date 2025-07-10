import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.functional as F



print("CUDA Available:", torch.cuda.is_available())
x = torch.rand(3, 3)
print(x)




'''
Creates a custom image Dateset
Images of insects 
'''
class IP102Dataset(Dataset):
    def __init__(self, split_file, image_root, transform=None):
        self.samples = []
        self.image_root = image_root
        self.transform = transform

        # Read split file
        with open(split_file, 'r') as f:
            for line in f:
                image_rel_path, label = line.strip().split()
                self.samples.append((image_rel_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_rel_path, label = self.samples[idx]
        image_path = os.path.join(self.image_root, image_rel_path)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
# Resize all images to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])    
'''
ToDO: make datasets for training and testing
'''
training_data = IP102Dataset(
    split_file="data/Ip102/raw/ip102_v1.1/train.txt",
    image_root="data/Ip102/raw/ip102_v1.1/images",
    transform=transform
)

test_data = IP102Dataset(
    split_file="data/Ip102/raw/ip102_v1.1/test.txt",
    image_root="data/Ip102/raw/ip102_v1.1/images",
    transform=transform
)


'''
Data loaders 
'''

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)



# Get a batch from the DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")  # e.g., [32, 3, 224, 224]
print(f"Labels batch shape: {train_labels.size()}")      # e.g., [32]

# Get the first image and label
img = train_features[0]
label = train_labels[0].item()  # convert from tensor to int

# Convert tensor to PIL image for display
plt.imshow(F.to_pil_image(img))  # no need for cmap
plt.axis('off')
plt.title(f"Label: {label}")
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 1. Load pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')  # Loads pretrained weights
model.fc = nn.Linear(model.fc.in_features, 102)   # Replace final layer
model = model.to(device)

# 2. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# 4. Evaluation loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return loss_total / total, accuracy

# 5. Train for a few epochs
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss:  {test_loss:.4f}, Accuracy: {test_acc:.2%}")