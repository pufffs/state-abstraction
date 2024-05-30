from Data_generation.DQN_agent import DQN
import torch
import numpy as np
from tqdm.notebook import tqdm

policy_net = DQN(8, 4)
model_path = "Data_generation/dqn_lunar_lander.pt"
policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #loaidng trained model
policy_net = policy_net.double()

def behavior_policy(state, epsilon=0):
    angle = state[2]
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice([0,1])
    else:
        if angle < 0:
            return 0
        else:
            return 1
        
def random_policy(state, action_size=2, batch=True):
    if batch:
        size = state.shape[0]
        return torch.randint(0, action_size, (size,)).long()
    
    return np.random.choice([0, 1])

def nondyna_policy(state, action=1, batch=True):
    if batch:
        size = state.shape[0]
        return torch.randint(action, action+1, (size,)).long()
    
    return action
def cartpole_policy(state, batch=True):
    if batch:
        pos = state[:,0]
        angle = state[:,2]
        prob_1 = 1 - 1/(1+torch.exp(angle-pos))
        return torch.bernoulli(prob_1).long()
    
    pos = state[0]
    angle = state[2]
    prob_0 = 1/(1+np.exp(angle-pos))
    prob_1 = 1 - prob_0
    return np.random.binomial(1, prob_1)

def angle_policy(state, batch=True):
    if batch:
        angle = state[:,2]
        return (angle>=0).long()
    
    return behavior_policy(state)

def target_lunar(state, batch=True):
    policy_net.eval()
    with torch.no_grad():
        if batch:
            a = torch.argmax(policy_net(state[:,:8]), axis=1).long()
            return a
        else:
            state = torch.tensor(state).double().view(1,-1)
            a = np.argmax(policy_net(state).squeeze(1).numpy())
            return a  

def behavior_lunar(state, epsilon=1):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice([0,1,2,3])
    else:
        return target_lunar(state, False)

def mc_oracle(env, num_episodes, target, gamma):
    rewards = np.zeros(num_episodes)
    state_size = env.observation_space.shape[0]
    for i in tqdm(range(num_episodes)):
        obs = env.reset()[0]
        done = False
        reward = 0
        gam_pow = 1
        count = 0
        while not done:
            a = target(obs, batch=False)
            next_obs, r, done, info, _ = env.step(a)
            if state_size==8:
                R = r
            else:
                R = adapt_r(obs)
            reward += gam_pow * R
            obs = next_obs
            gam_pow *= gamma
            if count>=1000 and state_size==8:
                break
        rewards[i] = reward
    mc_rewards_target_policy = np.mean(np.array(rewards))
    return mc_rewards_target_policy

def adapt_r(obs):
    return (1 - 5*obs[0]**2-10*obs[2]**2 )