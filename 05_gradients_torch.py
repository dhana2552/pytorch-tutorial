import torch

X = torch.tensor([1.0, 4.0, 3.0], dtype=torch.float32)
Y = torch.tensor([2.0, 8.0, 6.0], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# y_hat = f(x) = w*x

#forward pass
def f(x):
    return x*w

#MSE (as we are dealing with arrays and not a single value). MSE = 1/n * sum( (y_hat - y)**2 )
def loss(y, y_hat):
    return ((y_hat-y)**2).mean()

#calculate gradient
#dloss/dw = (2*x*(y_hat - y)).mean()

def grad(x, y, y_hat):
    return (2*x*(y_hat-y)).mean()

print(f'Prediction before training: f(3) = {f(3):.3f}')

#Training loop
learning_rate = 0.01
n_iter = 50

for epoch in range(n_iter):
    #prediction
    y_pred = f(X)
    #loss
    l = loss(Y, y_pred)
    #Backward pass
    l.backward() #dl/dw

    #update weights
    with torch.no_grad():
        w -= learning_rate*w.grad
    
    #zero gradients
    w.grad.zero_()
    
    if epoch%5 == 0:
        print(f'epoch: {epoch+1}, w = {w:.3f}, loss = {l:.3f}')
        
print(f'Prediction after training: f(3) = {f(3):.3f}')