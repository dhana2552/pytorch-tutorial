import torch

x = torch.rand(3, requires_grad=True)
y = x + 2
print(y)
z = y*y*2
v = torch.tensor([0.1, 1, 0.001], dtype=torch.float32) #gradient vector
z.backward(v) #dz/dx
print(z)

#to change gradience
#x.requires_grad_(False)
#x.detach()
#with torch.no_grad():

#x.requires_grad_(False)
x.requires_grad_(False)
print(x)

#x.detach()
y = y.detach()
print(y)

#with torch.no_grad():
with torch.no_grad():
    z = z+2
    print(z)
    
weights = torch.ones(3, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()