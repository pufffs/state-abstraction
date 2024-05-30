import torch
import copy

def encode(data, model, target_policy=None):
    #Encode the original (s,a,r,s') tuple by forward abstraction
    #data = [s,a,r,s',terminal] or [x,a,r,x',terminal,s,s']
    data = copy.deepcopy(data)
    sample_s = torch.cat([data[0], data[3][-1].unsqueeze(0)])
    model.eval()
    if target_policy is None:
        with torch.no_grad():
            all_states = model.encode(sample_s)
            phi_s = all_states[:-1]
            phi_s_next = all_states[1:]
    else: 
        if len(data)==5:
            pi_s = target_policy(sample_s)
        else:
            pi_s = target_policy( torch.cat([data[5], data[6][-1].unsqueeze(0)]) )
        with torch.no_grad():
            all_states = model.forward_encode(sample_s, pi_s)
            phi_s = all_states[:-1]
            phi_s_next = all_states[1:]
    data[0] = phi_s
    data[3] = phi_s_next
    return data
        
def data_format(data):
    s = torch.stack([torch.tensor(i[0]) for i in data]).double()
    a = torch.tensor([i[1] for i in data]).long()
    r = torch.tensor([i[2] for i in data])
    s_next = torch.stack([torch.tensor(i[3]) for i in data]).double()
    terminal = torch.stack([torch.tensor(i[4], dtype=torch.long) for i in data])
    return [s,a,r,s_next,terminal]

def load_combs(path):
    with open(path, 'rb') as f:
        dics = pickle.load(f)
        MSEs = torch.tensor(list(dics[0].keys())).numpy()
        sort_index = np.argsort(MSEs)
        sort_MSEs = MSEs[sort_index]
        sort_combs = torch.tensor(list(dics[0].values()))[sort_index]
        sort_bias = torch.tensor(list(dics[2].values()))[sort_index]
    return sort_MSEs, sort_combs, sort_bias