import torch
import torch.nn as nn
import torch.nn.functional as F

#define cnn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #try changing output to smaller into bigger
        self.conv1 = nn.Conv2d(3, 32, 7, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 7, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3,padding=1)
        #bn
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(512)
        #pool
        self.pool = nn.MaxPool2d(2, 2)
        #AdaptiveAvgPool2d
        self.aap = nn.AdaptiveAvgPool2d(6) 
        #dropout
        self.dropout = nn.Dropout(p=0.2) 
        #FCLs
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 106)

    def forward(self, x):
        #downstack
        x = self.downStack(x)
        #big pool
        x = self.aap(x)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        #FCLs
        #1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #3/output
        x = self.fc3(x) 
        return x
    
    def downStack(self, x):
        #conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.bn2(x)
        #pool
        x = self.pool(x)
        return x

   #apply soft max when only if not training
    def umm(self, x):
        # at runtime when not training us function in inference 
        return nn.Softmax(dim=1)(self.forward(x)) 