import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as Fu
from Model import Net
from DataSet import test_dataloader


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

# Move model to eval mode and correct device
net.eval()
net.to(device)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    pbar = tqdm(
        test_dataloader,
        desc=f"Testing: ",
        unit="batch",
    )
    for i, data in enumerate(pbar, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net.umm(inputs)  # forward 
    
        # print acc
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({
            "acc": 100. * correct / total
        })