#1. Design the model (input, output, forward pass)
#2. Construct loss and optimizer
#3. Train the model
# - forward pass: compute prediction
# - backward pass: compute gradient
# - update weights

import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

#0. Preprocessing data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#1. Build Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)
    
#2. Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3. Training
num_epochs = 100
for epoch in range(num_epochs):
    #predict
    y_pred = model(X_train)
    #forward pass
    l = criterion(y_pred, y_train)
    #backward pass
    l.backward()
    #update weights
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {l.item():.3f}')
        
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_numpy = y_pred.numpy()
    y_pred_numpy[y_pred_numpy >= 0.5] = 1
    y_pred_numpy[y_pred_numpy < 0.5] = 0
    print('Test Accuracy:', np.sum(y_pred_numpy == y_test.numpy())/len(y_test.numpy()))

