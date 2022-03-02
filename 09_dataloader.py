import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv', delimiter=',', skiprows=1, dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) #n_samples, 1
        self.n_samples = xy.shape[0]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
datatiter = iter(dataloader)
data = datatiter.next()
features1, labels1 = data
print(features1, labels1)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')