import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preparing data
bc = datasets.load_breast_cancer()
X , y = bc.data, bc.target                
''' 
    'X' : Input => "FEATURES"
    'y' : Output => "TARGET LABELS"

    'n_samples' : No.of samples in X  (e.g: S1, S2, ..., Sn)
    'n_features' : No. of features of X  (e.g: meanRadius, meanLength, meanTexture, etc...)
'''
n_sample, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.2)

    # Scaling features.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

    # Converting to torch tensors.
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

    # Converting "y" from ROWVECTOR -> COLUMNVECTOR
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1)Model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features ,1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred 
    
model = LogisticRegression(n_features)

# 2) Loss and optimizer.
learning_rate = 0.01

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training Loop.
num_epochs = 100
for epoch in range(num_epochs):
    # Forward Pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # Backward Pass
    loss.backward()
    # Updates
    optimizer.step()
    optimizer.zero_grad()

    if((epoch+1)%10 == 0):
        print(f'epoch: {epoch+1} loss: {loss.item():.4f}')

# We dont want evaluation to be part of
# our computational graph where, history is tracked.
with torch.no_grad():
    # Calculating accuracies by getting all the predicted classes
    # from our test.
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()

    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc.item():.4f}')