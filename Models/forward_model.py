import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.encoder import Encoder_linear, Encoder_conv

class Transition(torch.nn.Module):
    def __init__(self, in_dim, out_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.action_size = action_size
        self.activation = activation
        self.out_dim = out_dim
        self.hidden_size = 2*n_nodes

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(self.activation)
        
        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            if i < n_layers-2:
                self.net.append(self.activation)
                if dropout:
                    self.net.append(nn.Dropout(0.2))

        self.T_net = nn.Sequential(*self.net)

        self.lstm = nn.LSTMCell(n_nodes, 2*n_nodes)
        self.fcs = [nn.Linear(2*n_nodes, out_dim, dtype=torch.float64) for i in range(action_size)]

        self.tanh = nn.Tanh()

        self.train()

    def forward(self, x):
        h_0 = torch.zeros(x.shape[0], self.hidden_size, dtype=torch.float64)
        c_0 = torch.zeros(x.shape[0], self.hidden_size, dtype=torch.float64)
        x = self.T_net(x)
        h_1, c_1 = self.lstm(x, (h_0, c_0))
        x_list = [fc(h_1) for fc in self.fcs]
        x = torch.stack(x_list, dim=1)
        #x = self.fc(h_1)
        return x

class Reward(torch.nn.Module):
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
                self.net.append(nn.Dropout(0.2))

        self.net.append(nn.Linear(n_nodes, action_size))
        self.reward_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x):
        x = self.reward_net(x)
        return x

class Decoder(torch.nn.Module):
    #one of the penalty we used in experiments
    def __init__(self, in_dim, out_dim, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.2))

        self.net.append(nn.Linear(n_nodes, out_dim))
        self.decoder_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x):
        x = self.decoder_net(x)
        return x

class FQE(torch.nn.Module):
    def __init__(self, in_dim, action_size, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        self.action_size = action_size
        self.activation = activation

        self.action_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, in_dim) )
        self.xa_net = nn.Linear(2*in_dim, in_dim)

        self.net = []
        self.net.append(nn.Linear(in_dim, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.2))

        self.net.append(nn.Linear(n_nodes, action_size))
        self.FQE_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, x, pi_s):
        pi_s = pi_s.double().view(-1,1)
        action_pi = self.action_net(pi_s)
        x = torch.cat([x, action_pi], axis=1)
        phi1_s = self.xa_net(x)
        x = self.FQE_net(phi1_s)
        return x
    
    def forward_encode(self, x, pi_s):
        pi_s = pi_s.double().view(-1,1)
        action_pi = self.action_net(pi_s)
        x = torch.cat([x, action_pi], axis=1)
        phi1_s = self.xa_net(x)
        return phi1_s
    
    
class Forward_model(torch.nn.Module):
    "An assemble model for training with above forward abstraction objectives"
    def __init__(self, obs_size, encoder_dim, action_size, encoder_layers=3,
                 transition_layers=3, reward_layers=5, FQE_layers=3, activation=nn.ReLU()):
        super().__init__()
        self.obs_size = obs_size
        self.encoder_dim = encoder_dim
        self.action_size = action_size

        self.encoder = Encoder_linear(obs_size, encoder_dim, encoder_layers, activation=activation)
        self.transition = Transition(encoder_dim, encoder_dim, action_size, transition_layers, activation=activation)
        self.reward = Reward(encoder_dim, action_size, reward_layers, activation=activation)
        self.FQE = FQE(encoder_dim, action_size, FQE_layers, activation=activation)
        
    def forward(self, inputs):
        s, pi_s = inputs
        x = self.encoder(s)
        x_t = self.transition(x)
        x_r = self.reward(x)
        x_f = self.FQE(x, pi_s)
        return x, x_r, x_t, x_f
    
    def forward_encode(self, s, pi_s):
        #this return phi1(s), it takes information from target policy and is Q-pi irrelevant
        x = self.encoder(s)
        phi1_s = self.FQE.forward_encode(x, pi_s)
        return phi1_s
    
    def encode(self, s):
        # this is the model-irrelevant representation
        return self.encoder(s)
