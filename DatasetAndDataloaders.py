import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # Data loading
        xy = np.loadtxt('/Users/rishabhmathur/Documents/Development/Machine Learning/PyTorch/wine.csv', delimiter=",", 
                        skiprows=1, dtype=np.float32)

        self.x = torch.from_numpy(xy[:, 1:]) # (n_samples, all columns except 0th)
        self.y = torch.from_numpy(xy[:, 0]) # (n_samples, 0th column)

        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # Allows indexing later.
        # Dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # Allows to call len(dataset).
        return self.n_samples
    
dataset = WineDataset()
'''
first_data = dataset[0]
features, labels = first_data
print(features, labels)
'''

'''
# Converting from object to iterator

dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)
'''
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

#dummy training loop.
## Defining hyperparameters
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i,(inputs, labels) in enumerate(train_loader):
        if(i+1)%5==0:
            print(f'epoch: {epoch+1}/{num_epochs} steps: {i+1}/{n_iterations}, inputs: {inputs.shape}')