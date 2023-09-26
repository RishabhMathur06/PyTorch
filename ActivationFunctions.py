import torch
import torch.nn as nn
import torch.nn.functional as F

# OPTION 1: Create nn.modules
class NeuralNet(nn.Module):
    def __init__(self, inputSize , HiddenSize):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputSize, HiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(HiddenSize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out
    
# OPTION 2: Use ACTIVATION FUNCTIONS directly into the forward pass.
class NeuralNet(nn.Module):
    def __init__(self, inputSize, HiddenSize):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputSize, HiddenSize)
        self.linear2 = nn.Linear(HiddenSize, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(x))