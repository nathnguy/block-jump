# Neural Network Architecture and Q-value Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# whether or not we try to use MPS acceleration
USE_MPS = True

class BlockNet(nn.Module):

    # input size: size of the game window (256x256)
    # output size: number of actions (jump, stay)
    def __init__(self):
        super(BlockNet, self).__init__()

        # network architecture
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=4, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(512, 2)

        # optimizer and loss for Q-training
        # self.optimizer = optim.Adam(self.parameters(), lr=LR)
        # self.loss = nn.MSELoss()

        # choose device
        self.device = torch.device("mps" if torch.has_mps and USE_MPS else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

