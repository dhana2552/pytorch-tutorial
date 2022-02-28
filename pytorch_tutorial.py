from numpy import dtype
import torch

x = torch.rand(3, 2)
print(x)
y = torch.rand(3, 2)
print(y)
print(y.add_(x))
z = x.view([-1, 3])
print(z.size())