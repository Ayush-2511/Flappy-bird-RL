import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(12,256)
        self.layer2 = nn.Linear(256,256)
        self.layer3 = nn.Linear(256,2)


    def forward(self,x):

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)

        return x