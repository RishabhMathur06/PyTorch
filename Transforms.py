import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):
    def __init__(self, transform = None):
        # Data loading
        xy = np.loadtxt('', delimiter=",", 
                        skiprows=1, dtype=np.float32)

        self.x = torch.from_numpy(xy[:, 1:]) 
        self.y = torch.from_numpy(xy[:, 0]) 

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
Dataset = WineDataset()
