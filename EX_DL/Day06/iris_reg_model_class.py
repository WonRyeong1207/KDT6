import torch.nn as nn
import torch.nn.functional as F


class IrisRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_layer = nn.Linear(3, 20)
        self.hidden_layer = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, 1)
        
        
    def forward(self, X):
        y = F.relu(self.input_layer(X))
        y = F.relu(self.hidden_layer(y))
        y = self.output_layer(y)
        
        return y