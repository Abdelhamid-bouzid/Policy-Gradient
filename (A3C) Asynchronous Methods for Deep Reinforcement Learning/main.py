import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Agent import Agent
from ActorCritic import ActorCritic
from SharedAdam import SharedAdam

if __name__ == '__main__':
    lr = 1e-4
    env_id     = 'CartPole-v0'
    n_actions  = 2
    input_dims = [4]
    N_GAMES    = 3000
    T_MAX      = 5
    
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    
    optim     = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]