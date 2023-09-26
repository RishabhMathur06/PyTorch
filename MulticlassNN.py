import torch
from torch import nn

''' SOFTMAX '''

class NeuralNet2(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize, numClasses)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # No softmax at the end.
        return out

model =  NeuralNet2(inputSize=28*28, hiddenSize=5, numClasses=3)
citerion = nn.CrossEntropyLoss()