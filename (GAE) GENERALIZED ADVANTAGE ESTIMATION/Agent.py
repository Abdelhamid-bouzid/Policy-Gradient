import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from PPOMemory import PPOMemory
from ActorCritic import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma       = gamma
        self.policy_clip = policy_clip
        self.n_epochs    = n_epochs
        self.gae_lambda  = gae_lambda

        self.actor       = ActorNetwork(n_actions, input_dims, alpha)
        self.critic      = CriticNetwork(input_dims, alpha)
        self.memory      = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state  = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist   = self.actor(state)
        value  = self.critic(state)
        action = dist.sample()

        probs  = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value  = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, values_arr,reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            # compute Generalized Advantage Estimation using the sequence S,A,R,S'.... : The formula is A_t = TD_t + gamma*lamda*A_(t+1)           
            advantage_arr = np.zeros(len(reward_arr), dtype=np.float32)    
            a_t      = 0   
            for k in range(len(reward_arr)-1,-1,-1):
                td        = reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k]
                a_t       = td + a_t*self.gae_lambda*self.gamma
                advantage_arr[k] = a_t
            
            # divide the sequence S,A,R,S'.... into random batches 
            for batch in batches:
                states    = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions   = T.tensor(action_arr[batch]).to(self.actor.device)
                values    = T.tensor(values_arr[batch]).to(self.actor.device)
                advantage = T.tensor(advantage_arr[batch]).to(self.actor.device)

                dist         = self.actor(states)
                new_probs    = dist.log_prob(actions)
                
                critic_value = T.squeeze(self.critic(states))
                
                actor_loss   = advantage*new_probs

                returns      = advantage + values
                critic_loss  = T.mean((returns-critic_value)**2)

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

