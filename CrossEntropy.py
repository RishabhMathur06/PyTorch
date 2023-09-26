import torch
import numpy as np
import torch.nn as nn

'''def CrossEntropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# If:-
#   1) Class 0 = [1,0,0]
#   2) Class 1 = [0,1,0]
#   3) Class 2 = [0,0,1]

Y = np.array([1, 0, 0])

# y_pred has probabilities.
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

# Calculating loss.
l1 = CrossEntropy(Y, Y_pred_good)
l2 = CrossEntropy(Y, Y_pred_bad)

print(f'Loss1: {l1:.4f}')
print(f'Loss2: {l2:.4f}')
'''
# PyTorch
loss = nn.CrossEntropyLoss()

# 1x3
Y = torch.tensor([0])

# 3x3
Y = torch.tensor([2, 0, 1])

#When no. of samples = 1
'''
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
'''
# When no. of samples are 3
Y_pred_good = torch.tensor([[0.1, 1.0, 3.1], [2.0, 1.0, 0.1], [0.1, 3.0, 1.0]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [0.5, 2.0, 0.3], [0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)

print(prediction1)
print(prediction2)
