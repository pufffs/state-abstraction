import numpy as np
from tqdm.notebook import tqdm

class HD_obs:
    def __init__(self, env, num_episodes, behavior, epsilon=0.3, aug_dim=10):
        self.env = env
        self.num_episodes = num_episodes
        self.behavior = behavior
        self.epsilon = epsilon
        self.aug_dim = aug_dim
        self.state_size = env.observation_space.shape[0]
    def get_hd_obs(self):
        sars = []
        sars_by_episode = []
        initial_states = []

        for i in tqdm(range(self.num_episodes)):
            obs = self.env.reset()[0]
            done = False
            initial = True
            cur_sar = []
            noise = np.zeros(self.aug_dim)

            while not done:
                a = self.behavior(obs, self.epsilon)
                noise = self.AR1_noise(obs, noise, aug_dim=self.aug_dim)
                obs_hd = np.concatenate((obs, noise))

                if initial:
                    initial_states.append(obs_hd)
                    initial = False
                next_obs, r, done, info,_ = self.env.step(a)
                if self.state_size==4:
                    adaptive_r = self.adapt_r(obs)
                else:
                    adaptive_r = r
                    
                next_noise = self.AR1_noise(next_obs, noise, aug_dim=self.aug_dim)
                next_obs_hd = np.concatenate((next_obs, next_noise))

                cur_sar.append([obs_hd, a, adaptive_r, next_obs_hd, done])

                obs = next_obs
                noise = next_noise
                if self.state_size==8 and len(cur_sar)>1000: #stop lunar lander if it falls in one episode too long
                    cur_sar[-1][4]=True
                    break
            sars_by_episode.append(cur_sar)

        sars = [item for cur_sar in sars_by_episode for item in cur_sar]
        return sars, sars_by_episode, initial_states
    
    def AR1_noise(self, obs, last, coef=1,aug_dim=10):
        pass
    
    def adapt_r(self, obs):
        pass
    
class cartpole_HD(HD_obs):
    def __init__(self, env, num_episodes, behavior, epsilon=0.3, aug_dim=10):
        super().__init__(env, num_episodes, behavior, epsilon,aug_dim)
    
    def adapt_r(self, obs):
        return (1 - 5*obs[0]**2-10*obs[2]**2 )
    
    def AR1_noise(self, obs, last, coef=1,aug_dim=10):
        mean = np.mean(obs)
        std = np.std(obs)
        noise = coef*last + np.random.normal(2*mean, 10*std, aug_dim)
        return noise
    
class lunar_HD(HD_obs):
    def __init__(self, env, num_episodes, behavior, epsilon=0.3, aug_dim=10):
        super().__init__(env, num_episodes, behavior, epsilon,aug_dim)
    
    def AR1_noise(self, obs, last, coef=0.99,aug_dim=10):
        mean = np.mean(obs)
        std = np.std(obs)
        noise = coef*last + np.random.normal(mean, std, aug_dim)
        return noise
