import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as Fu
from Model import Net
from DataSet import train_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
#model
PATH = './cifar_net.pth'

# Check if file is there
if os.path.exists(PATH):
    print("✅ Model accessed successfully.")
else:
    print("❌ Failed.")

#load saved model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
net.to(device)
net.train()
print(net)
print(next(net.parameters()).device)  # should say: cuda:0

# Preview a batch and save image
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0]
label = train_labels[0].item()

plt.imshow(Fu.to_pil_image(img))
plt.axis('off')
plt.title(f"Label: {label}")
plt.savefig("sample_batch.png")
print("Saved a sample image to 'sample_batch.png'.")

# Model setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.00001)

# Fintune Model loop
EPOCHS = 5
for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0.0
    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        unit="batch",
    )
    for i, data in enumerate(pbar, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)  # forward 
        loss = criterion(outputs, labels) #  backward
        loss.backward() # optimize
        optimizer.step() # update weights

        running_loss += loss.item()
       
        # print acc
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({
            "loss": running_loss / (i + 1),
            "acc": 100. * correct / total
        })

print('Finished Training')

#save CNN
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Check if file was created
if os.path.exists(PATH):
    print("✅ Model saved successfully.")
else:
    print("❌ Save failed.")

