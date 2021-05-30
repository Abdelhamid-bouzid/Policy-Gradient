import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        
        self.local_actor_critic  = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name                = 'w%02i' % name
        self.episode_idx         = global_ep_idx
        self.env                 = gym.make(env_id)
        self.optimizer           = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done        = False
            observation = self.env.reset()
            score       = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action  = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(),self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                        
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                    
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)