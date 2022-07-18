import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        self.fc1 = nn.Linear(32 * 48 * 48, k)
    
    def forward(self, x):
        x = self.pool(F.relu(self.l1(x)))
        x = self.pool(F.relu(self.l2(x)))
        x = x.reshape(-1, 32*48*48)
        x = self.fc1(x)
        return x