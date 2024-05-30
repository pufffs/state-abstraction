import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, in_dim, action_size):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, obs_size, action_size, batch_size=64, n_layers=3):
        # general info
        self._state_size = obs_size 
        self._action_size = action_size
        self._batch_size = batch_size
        self.criterion_DQN = nn.MSELoss()
        # allow large replay exp space
        self._experience = deque(maxlen=1000000)
        
        self._gamma = 0.99
        # initialize with high exploration, which will decay later
        self._epsilon = 1.0 
        
        # Build Q Network
        self._policy_network = DQN(obs_size, action_size, n_layers).double()
        
        # Build Target Network - Build clone of Q Network
        self._target_network = copy.deepcopy(self._policy_network)
        
        self.optimizer = optim.Adam(self._policy_network .parameters(), lr=0.0001)
        
        self.update_target_network()
    
    # add new experience to the replay exp
    def memorize_experience(self, state, action, reward, next_state, done):
        self._experience.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        return self._target_network.load_state_dict(self._policy_network.state_dict())
    
    def choose_action(self, state):
        if np.random.rand() < self._epsilon: # exploration
            action = np.random.choice(self._action_size)
        else:
            state = torch.tensor(state).view(1,-1).double()
            self._policy_network.eval()
            with torch.no_grad():
                qhat = self._policy_network(state) # output Q(s,a) for all a of current state
            action = torch.argmax(qhat[0])
            action = int(action)
            
        return action
    
    def target_policy(self, state):
        self._policy_network.eval()
        with torch.no_grad():
            qhat = self._policy_network(state)
        return torch.argmax(qhat, axis=1)
     
    def experience_replay(self):
        # take a mini-batch from replay experience
        self.optimizer.zero_grad()
        self._policy_network.train()
        
        batch_size = min(len(self._experience), self._batch_size)
        mini_batch = random.sample(self._experience, batch_size)
        mini_batch = data_format(mini_batch)
        order = torch.arange(batch_size)
        
        sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones  = mini_batch
        self._target_network.eval()
        with torch.no_grad():   
            sample_qhat_next = self._target_network(sample_next_states)
            
        sample_qhat = self._policy_network(sample_states).gather(1, sample_actions.unsqueeze(1)).squeeze(1)
        
        #sample_qhat_targets = sample_qhat.detach().clone()
        # set all Q values terminal states to 0
        sample_qhat_targets = sample_rewards + self._gamma * torch.amax(sample_qhat_next, axis=1) * (torch.ones(batch_size) - sample_dones)
        
        loss_DQN = self.criterion_DQN(sample_qhat, sample_qhat_targets.detach())

        loss_DQN.backward()
        self.optimizer.step()
        
        return loss_DQN.item()