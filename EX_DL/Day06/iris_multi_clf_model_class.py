import torch.nn as nn
import torch.nn.functional as F

class IrisMCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(4, 10)
        self.hidden_layer = nn.Linear(10, 5)
        self.output_layer = nn.Linear(5, 3)
        
    def forward(self, X):
        
        y = F.relu(self.input_layer(X))
        y = F.relu(self.hidden_layer(y))
        y = self.output_layer(y)
        
        return y