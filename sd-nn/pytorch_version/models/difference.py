import torch.nn as nn
import torch.nn.functional as F


# definition of the Coarse Model
class DifferenceModel(nn.Module):
    """main neural network to learn input/position relationship"""
    def __init__(self, n_features, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x))
        out = self.fc2(out)
        return out
