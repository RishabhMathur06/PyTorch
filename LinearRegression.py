import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

''' 
    Steps to be followed:- 
        1. Design a model
        2. Loss and optimizer
        3. Training loop
'''

# Step:0
# Prepare data.
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Converting to tensor.
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))     # Converting from "DOUBLE" -> "FLOAT 32"

# Reshaping "y" into column vector.
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# Step:1
# Designing a model.
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)
### NOTE: Here, we are using a single layer model.

# Step:2
# Calculating loss and optimization.
learning_rate = 0.01
criterion = nn.MSELoss()        # In case of Linear Regression: loss = M.S.E
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step:3
# Training loop.
num_epochs = 100
for epoch in range(num_epochs):
    # Perform:-
    #   1. Forward Pass
    #   2. Backward Pass
    #   3. Update Weights

    ''' Forward Pass and Loss'''
    y_pred = model(X)
    loss = criterion(y_pred, y)

    ''' Backward Pass '''
    # Calculating gradients
    loss.backward()

    ''' Update weights '''
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1} loss: {loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()

'''
    Here, we run for the final "model" and don't want graph to track this operation
    therefore, we use "detach()".

    This makes "requires_grad = False"

    And, convert it into numpy to plot it.
'''
plt.plot(X_numpy, y_numpy, 'ro')
# Plot generateed function
plt.plot(X_numpy, predicted, 'b')
plt.show()