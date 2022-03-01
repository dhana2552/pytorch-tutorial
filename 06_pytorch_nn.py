#1. Design the model (input, output, forward pass)
#2. Construct loss and optimizer
#3. Train the model
# - forward pass: compute prediction
# - backward pass: compute gradient
# - update weights

import torch
import torch.nn as nn

X = torch.tensor([[1.0], [4.0], [3.0]], dtype=torch.float32)
Y = torch.tensor([[2.0], [8.0], [6.0]], dtype=torch.float32)

X_test = torch.tensor([[5.0]], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: X_test = 5: {model(X_test).item():.3f}')

#Training loop
learning_rate = 0.01
n_iter = 500

#MSE (as we are dealing with arrays and not a single value). MSE = 1/n * sum( (y_hat - y)**2 )
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iter):
    #prediction
    y_pred = model(X)
    #loss
    l = loss(Y, y_pred)
    #Backward pass
    l.backward() #dl/dw

    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if epoch%50 == 0:
        [w, b] = model.parameters()
        print(f'epoch: {epoch+1}, w = {w[0][0].item():.3f}, loss = {l:.3f}')
        
print(f'Prediction after training: X_test = 5: {model(X_test).item():.3f}')