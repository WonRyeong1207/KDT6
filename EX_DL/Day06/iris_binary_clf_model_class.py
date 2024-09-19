import torch.nn as nn
import torch.nn.functional as F

class IrisBCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(4, 20)
        self.hidden_layer = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, 1)
    
    def forward(self, X):
 
        y = F.relu(self.input_layer(X))
        y = F.relu(self.hidden_layer(y))
        y = F.sigmoid(self.output_layer(y))
        
        return y