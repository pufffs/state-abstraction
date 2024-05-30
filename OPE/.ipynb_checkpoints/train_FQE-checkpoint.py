from OPE.FQE import FQE_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset

def train_FQE(data, num_epochs, target_policy, n_layers=3, n_nodes=32, lr=0.001):  
    #data = [x,a,r,x',terminal,s']
    action_size =len(torch.unique(data[1]))
    obs_size = data[0][0].shape[0]
    model = FQE_eval(obs_size, action_size, n_layers, n_nodes).double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    terminal_index = torch.nonzero(data[4]).squeeze().long() #the next index of True terminal is the initial state of the next episode
    terminal_index = terminal_index[:-1] #remove the last terminal state index
    init_index = torch.cat([torch.tensor([0]), terminal_index+1]) #the first state is always initial
    observed_init_index = torch.cat([torch.tensor([0]), terminal_index]) #use s_next to get initial observed states
    initial_x = data[0][init_index]
    observed_init = data[5][observed_init_index]
    target_init = target_policy(observed_init) 
    num_episode = initial_x.shape[0]
    
    batch_size = max((data[1].shape[0])//20,10)
    dataset = TensorDataset(*data)
    batch_data = DataLoader(dataset, batch_size=batch_size)
    
    for epoch in tqdm(range(num_epochs)):
        for x, a, r, x_next, terminal, observed_s_next in batch_data:
            batch_loss = train_FQE_step(model, optimizer, x, a, r, x_next, terminal, observed_s_next, target_policy)
            
        model.eval()
        with torch.no_grad():
            preds = model(initial_x) #Q-value estimation is based on abstracted space
        estimated_value = preds[np.arange(num_episode), target_init]
        estimated_value = estimated_value.mean()
    return estimated_value
    
def train_FQE_step(model, optimizer, x, a, r, x_next, terminal, observed_s_next, target_policy, gamma=0.99):
    optimizer.zero_grad()
    model.train()
    criterion_FQE = nn.MSELoss()

    batch_size = x.shape[0]
    order = torch.arange(batch_size)
    pi_s_next = target_policy(observed_s_next)  #the policy is based on observed state space

    outputs_FQE = model(x)
    with torch.no_grad():
        FQE_next = model(x_next)

    FQE_targets = outputs_FQE.detach().clone()

    FQE_targets[order, a] = r + gamma * FQE_next[order, pi_s_next] * (torch.ones(batch_size) - terminal)

    loss_FQE = criterion_FQE(outputs_FQE, FQE_targets)

    loss_FQE.backward()
    optimizer.step()

    return loss_FQE.item()
    