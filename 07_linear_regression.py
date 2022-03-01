#1. Design the model (input, output, forward pass)
#2. Construct loss and optimizer
#3. Train the model
# - forward pass: compute prediction
# - backward pass: compute gradient
# - update weights

from pickletools import optimize
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0. Preprocessing data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
print(X_numpy.shape, y_numpy.shape)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
y = y.view(y.shape[0], 1)

print(X.shape, y.shape)

#1. Build Model
n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#2. Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3. Training
num_epochs = 100
for epoch in range(num_epochs):
    #predict
    y_pred = model(X)
    #forward pass
    l = criterion(y, y_pred)
    #backward pass
    l.backward()
    #update weights
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {l.item():.3f}')
        
        
#4. Visualization
prediction = model(X).detach().numpy() #detach to get rid of gradients
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, prediction, 'b')
plt.show()
