import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.encoder import Encoder_linear, Encoder_conv

class Inverse(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(2*in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.3))

        if action_size == 2:
            self.net.append(nn.Linear(n_nodes, 1))
            #self.net.append(nn.Sigmoid())
        else:
            self.net.append(nn.Linear(n_nodes, action_size))

        self.inverse_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x, x_next):
        x = torch.cat([x, x_next], -1)
        x = self.inverse_net(x)
        return x


class Density(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(2*in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.3))

        self.net.append(nn.Linear(n_nodes, 1))

        self.density_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x, x_next):
        x_next_flipped = torch.flip(x_next, dims=[0])
        x_pos = torch.cat([x, x_next], -1)
        x_neg = torch.cat([x, x_next_flipped], -1)
        x = torch.cat([x_pos, x_neg], 0)
        
        x = self.density_net(x)
        x = x.flatten()
        return x

class Rho(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.action_size = action_size
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.3))

        self.net.append(nn.Linear(n_nodes, action_size))
        self.rho_net = nn.Sequential(*self.net)
        
        self.train()
    def forward(self, x):
        x = self.rho_net(x)
        return x

class Backward_model(torch.nn.Module):
    "An assemble model for training with above backward abstraction objectives"
    def __init__(self, obs_size, encoder_dim, action_size, encoder_layers=3,
                 inverse_layers=5, density_layers=5, rho_layers=3):
        super().__init__()
        self.obs_size = obs_size
        self.encoder_dim = encoder_dim
        self.action_size = action_size

        self.encoder = Encoder_linear(obs_size, encoder_dim, encoder_layers)
        self.inverse = Inverse(encoder_dim, action_size, inverse_layers)
        self.density = Density(encoder_dim, action_size, density_layers)
        self.rho = Rho(encoder_dim, action_size, rho_layers)
    def forward(self, inputs):
        s, s_next = inputs
        x = self.encoder(s)
        x_next = self.encoder(s_next)
        x_i = self.inverse(x, x_next)
        x_d = self.density(x, x_next)
        x_rho = self.rho(x)
        return x, x_next, x_i, x_d, x_rho
    
    def encode(self, s):
        #this return phi2(s), the backward model irrelevant representation
        phi2_s = self.encoder(s)
        return phi2_s

class Behavior(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.action_size = action_size
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.3))
                
        if action_size == 2:
            self.net.append(nn.Linear(n_nodes, 1))
        else:
            self.net.append(nn.Linear(n_nodes, action_size))
            
        self.behavior_net = nn.Sequential(*self.net)
        
        self.train()
    def forward(self, s):
        x = self.behavior_net(s)
        return x