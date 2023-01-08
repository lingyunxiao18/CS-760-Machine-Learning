import numpy as np
import torch 
from torch import distributions
from torch import nn
from torch.utils import data

class classifier(nn.Module):
    def __init__(self, dim):
        super(classifier, self).__init__()
#        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),                 
            nn.Linear(64, 64),            
            nn.ReLU(),                 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x).squeeze()
        return logits

    def focalloss(self,y1,y2,w=10):
      
      l= - w * y2*torch.log(1e-8+y1) +(y2-1)*torch.log(1e-8+1- y1)
      return torch.mean(l)

    def train(self, X, y, iter = 500, lr=1e-4, loss= None, batch_size = 64):
      
        if loss is None:
          loss= nn.BCELoss()
        elif loss==1:
          loss =self.focalloss

        training_set=data.TensorDataset(torch.Tensor(X),torch.Tensor(y))
        trainloader = data.DataLoader(dataset=training_set, batch_size = batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        losses = []
        t = 0 # iteration count
        while t < iter:
            for i,(X,y) in enumerate(trainloader):  
                outputs=self.forward(X)
                l=loss(outputs,y)
                losses.append(l.item())
                optimizer.zero_grad()
                l.backward(retain_graph=True)
                optimizer.step()
                t=t+1
        return losses

    def predict(self, X):
        logits=self.forward(torch.Tensor(X)).detach().numpy()
        result=(logits>0.5).astype(int)
        return result
    
class regression(nn.Module):
    def __init__(self, dim):
        super(regression, self).__init__()
#        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),      
            nn.Linear(64, 64),
            nn.ReLU(),   
            nn.Linear(64, 64),
            nn.ELU(),                    
            nn.Linear(64, 1),
        )
    def forward(self, x):
        y = self.linear_relu_stack(x).squeeze()
        return y
    
    
    def train(self, X, y, iter = 500, lr=1e-4, loss= None, batch_size = 64):
      
        if loss is None:
          loss= nn.MSELoss()

        training_set=data.TensorDataset(torch.Tensor(X),torch.Tensor(y))
        trainloader = data.DataLoader(dataset=training_set, batch_size = batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        losses = []
        t = 0 # iteration count
        while t < iter:
            for i,(X,y) in enumerate(trainloader):  
                outputs=self.forward(X)
                l=loss(outputs,y)
                losses.append(l.item())
                optimizer.zero_grad()
                l.backward(retain_graph=True)
                optimizer.step()
                t=t+1
        return losses

    def predict(self, X):
        y=self.forward(torch.Tensor(X)).detach().numpy()
        return y