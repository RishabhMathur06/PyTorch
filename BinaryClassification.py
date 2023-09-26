import torch
from torch import nn

''' SOFTMAX '''

class NeuralNet1(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # Sigmoid at the end.
        y_pred = torch.sigmoid(out)
        return y_pred

model =  NeuralNet1(inputSize=28*28, hiddenSize=5)
citerion = nn.BCELoss()