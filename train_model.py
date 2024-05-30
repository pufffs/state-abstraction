import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
import copy
from Models.losses import SmoothLoss, CollapseLoss
from Models.encoder import Encoder_linear, Encoder_conv
from Models.forward_model import Forward_model
from Models.backward_model import Backward_model, Behavior
from Utils.util_functions import encode

# Functions for training forward models
def train_forward(data, target_policy, params={}):
    #data = [s,a,r,s',terminal] or [x,a,r,x',terminal,s,s']
    obs_size = data[0][0].shape[0]
    action_size = params.get("action_size", 2)
    encoder_dim = params.get("encoder_dim", 10)
    encoder_layers = params.get("encoder_layers", 3)
    transition_layers = params.get("transition_layers", 3)
    reward_layers = params.get("reward_layers", 5)
    FQE_layers = params.get("FQE_layers", 3)
    lr = params.get("lr", 0.001)
    num_epochs = params.get("num_epochs", 30)
    alphar = params.get("alpha_r",1.)
    alphat = params.get("alpha_t",1.)
    alphaf = params.get("alpha_f",1.)
    #alphap = params.get("alpha_p",1.)
    alphap = min(1, 10/obs_size)
    
    observed = False if len(data)==5 else True
    model = Forward_model(obs_size, encoder_dim, action_size, encoder_layers,
                     transition_layers, reward_layers, FQE_layers).double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    batch_size = max((data[1].shape[0])//20,10)
    dataset = TensorDataset(*data)
    batch_data = DataLoader(dataset, batch_size=batch_size)
    gen_len = len(batch_data)
    epoch_loss = []
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = np.zeros(4)
        for batch in batch_data:
            if observed:
                s, a, r, s_next, terminal, observed_s, observed_next = batch
            else:
                s, a, r, s_next, terminal = batch
                observed_s = s.clone().detach()
                observed_next = s_next.clone().detach()
                
            batch_loss = train_step(model, optimizer, s, a, r, s_next,terminal,
                            observed_s, observed_next, target_policy, alphar, alphat, alphaf, alphap)

            running_loss += np.array(batch_loss)
        epoch_loss.append(running_loss / gen_len )
    return model, epoch_loss
    
def train_step(model, optimizer, s, a, r, s_next, terminal, observed_s, observed_next, target_policy,
               alpha_r=1., alpha_t=1., alpha_f=1., alpha_p=0.1):
    optimizer.zero_grad()
    model.train()
    
    mse_loss = nn.MSELoss()
    smooth_loss = SmoothLoss()
    collapse_loss = CollapseLoss()
    
    batch_size = s.shape[0]
    order = torch.arange(batch_size)
    observed_s = observed_s.requires_grad_()
    pi_s = target_policy(observed_s)
    pi_s_next = target_policy(observed_next)

    outputs_encoder, outputs_reward, outputs_transition, outputs_FQE = model((s, pi_s))

    with torch.no_grad():
        model.eval()
        x_next = model.encoder(s_next)
        FQE_next = model.FQE(x_next, pi_s_next)
        x_next_hat = model.transition(x_next)
    model.train()
    
    x_next_flip = torch.flip(x_next, dims=[0])
    x_next_hat_flip = torch.flip(x_next_hat, dims=[0])
    
    # Compute the losses
    x_next_targets = outputs_transition.detach().clone()
    x_next_targets[order, a] = x_next

    FQE_targets = outputs_FQE.detach().clone()
    FQE_targets[order, a] = r + .99 * FQE_next[order, pi_s_next] *\
                    (torch.ones(batch_size, dtype=torch.float64) - terminal)

    r_targets = outputs_reward.detach().clone()
    r_targets[order, a] = r

    loss_reward = mse_loss(outputs_reward, r_targets)
    loss_transition = mse_loss(outputs_transition, x_next_targets)
    loss_FQE = mse_loss(outputs_FQE, FQE_targets)
    loss_collapse1 = collapse_loss(outputs_encoder, x_next_flip) #by revsersing the order, the loss is calculated on random samples of states
    loss_collapse2 = collapse_loss(x_next_hat_flip, outputs_transition)
    loss_smooth = smooth_loss(x_next, outputs_encoder) #if two states are consective then they should be closer in abstract state space as well
    
    loss_penalty = loss_collapse1 + loss_collapse2 + loss_smooth

    l_phi_1 = alpha_r*loss_reward + alpha_t*loss_transition + alpha_f*loss_FQE + alpha_p*loss_penalty
    
    l_phi_1.backward()
    optimizer.step()
    
    return alpha_r*loss_reward.item(), alpha_t*loss_transition.item(), alpha_f*loss_FQE.item(), alpha_p*loss_penalty.item()


# Functions for training backward models
def train_backward(data, target_policy, params={}):
    #data=[s,a,r,s',terminal] or [x,a,r,x',terminal,s,s']
    obs_size = data[0][0].shape[0]
    action_size = params.get("action_size", 2)
    encoder_dim = params.get("encoder_dim", 10)
    encoder_layers = params.get("encoder_layers", 3)
    inverse_layers = params.get("inverse_layers", 3)
    density_layers = params.get("density_layers", 3)
    rho_layers = params.get("rho_layers", 3)
    
    alphai = params.get("alpha_i",1.)
    alphad = params.get("alpha_d",1.)
    alpharho = params.get("alpha_rho",1.)
    #alpharho = min(1., 20/obs_size)
    alphap = min(1., 10/obs_size)
    #alphap = params.get("alpha_p",1.)

    lr = params.get("lr", 0.001)
    num_epochs = params.get("num_epochs", 30)
    #encoder_gap = params.get("encoder_gap", 3)
    observed = False if len(data)==5 else True

    behavior_fitted, b_loss = train_behavior(data[:2], 30, action_size)
        
    model = Backward_model(obs_size, encoder_dim, action_size, encoder_layers,
                     inverse_layers, density_layers, rho_layers).double()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    batch_size = max((data[1].shape[0])//20,10)
    dataset = TensorDataset(*data)
    batch_data = DataLoader(dataset, batch_size=batch_size)
    gen_len = len(batch_data)
    epoch_loss = []
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = np.zeros(4)
        for batch in batch_data:
            if observed:
                s, a, _, s_next, _, observed_s,_ = batch
            else:
                s, a, _, s_next, _= batch
                observed_s = s.clone().detach()
                
            batch_loss = back_train_step(model, behavior_fitted, optimizer, s, a, s_next, observed_s,
                                         target_policy, action_size, alphai, alphad, alpharho, alphap)

            running_loss += np.array(batch_loss)
        epoch_loss.append(running_loss / gen_len )
    return model, epoch_loss
    
def back_train_step(model, behavior_model, optimizer, s, a, s_next, observed_s, target_policy, 
                   action_size, alpha_i=1., alpha_d=1., alpha_rho=1., alpha_p=0.5):
    optimizer.zero_grad()
    model.train()
    
    mse_loss = nn.MSELoss()
    smooth_loss = SmoothLoss()
    collapse_loss = CollapseLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    batch_size, s_size = s.shape
    order = torch.arange(batch_size)
    
    pi_a_s = (target_policy(observed_s)==a).double() #the probability of target policy taking action a.
    behavior_model.eval()
    with torch.no_grad():
        b_a_s = behavior_model(s) #b_a_s computational graph breaken
        
    if action_size > 2:
        b_a_s = torch.softmax(b_a_s, dim=1)
        b_a_s = b_a_s[order, a.long()] 
        rho_a_s = (pi_a_s/b_a_s)
        
        inv_loss = nn.CrossEntropyLoss()
    else:
        b_a_s = torch.sigmoid(b_a_s).view(-1,)
        for i in range(batch_size):
            if not a[i]:
                b_a_s[i] = 1. - b_a_s[i]
        rho_a_s = (pi_a_s/b_a_s)
        
        a = a.double()
        inv_loss = bce_loss
        
    s_next_flipped = torch.flip(s_next, dims=[0])
    contrastive_labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)])
    
    x, x_next, x_inverse, x_density, x_rho = model((s, s_next))
    
    x_next_flip = torch.flip(x_next, dims=[0])
    x_rho_targets = x_rho.detach().clone()
    x_rho_targets[order, a.long()] = rho_a_s
    if action_size == 2:
        a = a.view(-1,1)
    loss_inverse = inv_loss(x_inverse, a)
    loss_density = bce_loss(x_density, contrastive_labels)
    loss_rho = mse_loss(x_rho, x_rho_targets)
    loss_collapse = collapse_loss(x, x_next_flip)
    loss_smooth = smooth_loss(x, x_next)
    loss_penalty = loss_collapse + loss_smooth

    l_phi_2 = alpha_i*loss_inverse + alpha_d*loss_density + alpha_rho*loss_rho + alpha_p *loss_penalty
    
    l_phi_2.backward()
    optimizer.step()
    
    return alpha_i*loss_inverse.item(), alpha_d*loss_density.item(), alpha_rho*loss_rho.item(), alpha_p*loss_penalty.item()

def train_behavior(data, num_epochs, action_size, n_layers=3, n_nodes=64, lr=0.001):
    #data=[s,a] or [x,a]
    in_dim = data[0][0].shape[0]
    if action_size > 2:
        b_loss = nn.CrossEntropyLoss()
    else:
        b_loss = nn.BCEWithLogitsLoss()
        
    batch_size = max((data[1].shape[0])//20,10)
    model = Behavior(in_dim, action_size, n_layers, n_nodes).double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(*data)
    batch_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen_len = len(batch_data)
    epoch_loss = []
    
    for epoch in range(num_epochs):
        running_loss = 0
        for s, a in batch_data:
            if action_size == 2:
                a = a.double()
                a = a.view(-1,1)
                
            optimizer.zero_grad()
            model.train()
            
            b_a_s = model(s)
            loss_behavior = b_loss(b_a_s, a)
            loss_behavior.backward()
            optimizer.step()
            
            running_loss += loss_behavior.item()
            
        epoch_loss.append(running_loss / gen_len )
    return model, epoch_loss



# Functions for training two-step models
def train_two_step(data, modelnames, target_policy, params):
    #data=[s,a,r,s',terminal]
    observed_s = data[0].detach().clone()
    observed_next = data[3].detach().clone()
    data = copy.deepcopy(data) 
    for i, model_name in enumerate(modelnames):
        model_params = params[i]
        if i==0:
            if model_name=="forward":
                
                forward_model, _ = train_forward(data, target_policy, model_params)
                data = encode(data, forward_model, target_policy)
            else:
                
                backward_model,_ = train_backward(data, target_policy, model_params)
                data = encode(data, backward_model)
            data.extend([observed_s, observed_next])
        else:
            if model_name=="forward":
                
                model, loss = train_forward(data, target_policy, model_params)
                data = encode(data, model, target_policy)
            else:
                
                model, loss = train_backward(data, target_policy, model_params)
                data = encode(data, model)
                
    return model, loss, data