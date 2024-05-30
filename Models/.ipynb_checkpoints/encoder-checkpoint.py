import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_linear(torch.nn.Module):
    def __init__(self, obs_size, out_dim, n_layers=2, n_nodes=64, activation=nn.ReLU(), dropout=True):
        super().__init__()
        assert n_layers >= 1
        self.obs_size = obs_size
        self.activation = activation

        self.net = []
        self.net.append(nn.Linear(obs_size, n_nodes))
        self.net.append(self.activation)

        for i in range(n_layers-1):
            self.net.append(nn.Linear(n_nodes, n_nodes))
            self.net.append(self.activation)
            if dropout:
                self.net.append(nn.Dropout(0.2))

        self.net.append(nn.Linear(n_nodes, out_dim))
        #self.net.append(nn.Tanh())
        
        self.encoder_net = nn.Sequential(*self.net)

        self.train()
    def forward(self, s):
        x = self.encoder_net(s)
        return x

class Encoder_conv(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)

        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, out_dim)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.train()
    def forward(self, s):
        x = self.conv1(s)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        # Skip pooling here due to dimension consideration

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.tanh(self.fc2(x))
        x = self.fc2(x)
        return x