import torch
import numpy as np
import torch.nn as nn

#with numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss
    
y = np.array([1, 0, 0])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

print('Good prediction cross entropy loss: ',cross_entropy(y, y_pred_good))
print('Bad prediction cross entropy loss: ',cross_entropy(y, y_pred_bad))

#with torch
loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
print(Y.shape)
#num_samples*num_classes = 1*3
y_pred_good = torch.tensor([[0.7, 0.2, 0.1]])
y_pred_bad = torch.tensor([[0.1, 0.3, 0.6]])
l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, prediction1 = torch.max(y_pred_good, 1)
_, prediction2 = torch.max(y_pred_bad, 1)
print(prediction1.item())
print(prediction2.item())

#with more samples
Y = torch.tensor([0, 1, 2])
y_pred_good = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.7, 0.6], [0.1, 0.3, 0.6]])
y_pred_bad = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, prediction1 = torch.max(y_pred_good, 1)
_, prediction2 = torch.max(y_pred_bad, 1)
print(prediction1)
print(prediction2)
