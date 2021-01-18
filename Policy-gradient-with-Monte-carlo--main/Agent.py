# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:20:14 2020

@author: Abdelhamid
"""
import torch as T
from PolicyNet import PolicyNet
import numpy as np
from torch.distributions import Categorical

class Agent(object):
    def __init__(self, lr, n_actions, input_dim, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01, max_iterations=1000, lamda=0.9):
        
        self.lamda        = lamda
        self.lr           = lr
        self.n_actions    = n_actions
        self.input_dim    = input_dim
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.eps_dec      = eps_dec
        self.eps_min      = eps_min
        self.max_iterations = max_iterations
        self.action_space = [i for i in range(self.n_actions)]
        
        self.Policy            = PolicyNet(self.lr, self.n_actions, self.input_dim)
        
    def choose_action(self, state):
        self.Policy.eval()
        if np.random.random()> self.epsilon:
            
            state  = T.tensor(state, dtype=T.float).to(self.Policy.device)
            probs  = self.Policy(state)
            c      = Categorical(probs)
            action = c.sample().detach().cpu().numpy()
            
        else:
            action = np.random.choice(self.action_space)
        self.Policy.train()
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                            
    def learn(self,env):
        
        self.Policy.optimizer.zero_grad()
        
        states, actions, rewards = self.play_episode(env)
        
        for i in range(len(states)):
        
            state   = T.tensor(states[i], dtype=T.float).to(self.Policy.device)
            action  = T.tensor(actions[i]).to(self.Policy.device)
            reward  = T.tensor(rewards[i]).to(self.Policy.device)
            
            probs  = self.Policy(state)
            c      = Categorical(probs)
            
            loss = -c.log_prob(action) * reward
            loss.backward()
            
        self.Policy.optimizer.step()
        self.decrement_epsilon()
            
        return sum(rewards), len(rewards)
        
    def play_episode(self, env):
    
        done   = False
        i      = 0
        state  = env.reset()
        
        rewards   = []
        states    = []
        actions   = []
        while (not done) or (i<self.max_iterations):
            
            action = self.choose_action(state)
            n_state, reward, done, info = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            
            state = n_state
            
            i +=1
            
        R = 0
        new_rewards = rewards
        for i in range(len(rewards)-1, -1, -1):
            R = (rewards[i] + self.gamma * R)
            new_rewards[i] = R
            
        #Normalize reward
        reward_mean = np.mean(new_rewards)
        reward_std  = np.std(new_rewards)
        new_rewards = (new_rewards-reward_mean)/reward_std
        
        return states, actions, new_rewards
    
    
    
    
    
    
