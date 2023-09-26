# MNIST

#Steps:-
'''
    1. Dataloader
    2. Transformation
    3. Multilayer Neural Net
    4. Activation Function
    5. Loss & Optimiszer
    6. Training loop
    7. Model Evaluation
    8. GPU support
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Device config.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters.
inputSize = 784  # Flattening this input 28*28 into tensor would be 784 size array.
hiddenSize = 500
num_classes = 10 # Since, we have digits 0-9
num_epochs = 2   # We don't want training to be too long.
batch_size = 100
learning_rate = 0.001

# Importing dataset: MNIST.
train_dataset = torchvision.datasets.MNIST(
    root = './data', 
    train=True, 
    transform=transforms.ToTensor(),
    download=True
)
 
test_dataset = torchvision.datasets.MNIST(
    root = './data', 
    train=False,
    transform=transforms.ToTensor()
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True   
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False   
)

examples = iter(train_loader)
samples, labels = next(examples)  #Unpacking
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

# Ouput: [100, 1, 28, 28] [100]

## 100-> Hidden Size
## 1 -> Color Channel = 1 since only gray color.
## 28*28 -> actual image array.
## 100 -> Each class label

# To classify these digits, we set up "FULLY CONNECTED NN"
class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out
        # Since, we don want to apply "Activation Function" at the end of network.

model = NeuralNet(inputSize, hiddenSize, num_classes)
# Loss & Optimizer.
criterion = nn.CrossEntropyLoss()       # "Softmax" is applied here.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshaping images as our shape is: [100, 1, 28, 28],
        # but we want it to be: [100, 784] as input should be flattened.
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward Pass:-
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass:-
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        # Updates parameters for us.

        if(i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
# Testing loop.
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)

        n_correct = (predictions == labels).sum().item()

    acc = 100.0*n_correct/n_samples
    print("Accuracy={} ".format(acc))

