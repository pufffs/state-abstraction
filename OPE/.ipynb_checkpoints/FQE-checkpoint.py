import torch
import torch.nn as nn
import torch.nn.functional as F

class FQE_eval(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=32, activation=nn.ReLU()):
        super().__init__()
        self.action_size = action_size

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(activation)

        self.net.append(nn.Linear(n_nodes, action_size))
        self.FQE_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x):
        x = self.FQE_net(x)
        return x