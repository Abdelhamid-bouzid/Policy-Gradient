# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:20:14 2020

@author: Abdelhamid
"""
import torch as T
from Actir_Policy import Actor
from Critic import Critic
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
        
        self.Actor        = Actor(self.lr, self.n_actions, self.input_dim)
        self.Critic       = Critic(self.lr, 1, self.input_dim)
        
    def choose_action(self, state):
        self.Actor.eval()
        if np.random.random()> self.epsilon:
            
            state  = T.tensor(state, dtype=T.float).to(self.Actor.device)
            probs  = self.Actor(state)
            c      = Categorical(probs)
            action = c.sample().detach().cpu().numpy()
            
        else:
            action = np.random.choice(self.action_space)
        self.Actor.train()
        
        self.Critic.eval()
        state      = T.tensor(state, dtype=T.float).to(self.Critic.device)
        s_a_value  = self.Critic(state)
        self.Critic.train()
        return action,s_a_value
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                            
    def learn(self,env):
        
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()
        
        states, actions, G = self.play_episode(env)
        actor_loss  = []
        critic_loss = []
        for i in range(len(states)):
        
            state   = T.tensor(states[i], dtype=T.float).to(self.Actor.device)
            action  = T.tensor(actions[i]).to(self.Actor.device)
            G_s_a   = T.tensor(G[i]).to(self.Actor.device)
            
            # actor probabilities over state s 
            probs  = self.Actor(state)
            c      = Categorical(probs)
            
            # critic evaluation of state s 
            s_pred  = self.Critic(state)
            
            advantage = (G_s_a - s_pred)
            actor_loss.append(-c.log_prob(action) * advantage)
            
            critic_loss.append(self.Critic.loss(G_s_a,s_pred).to(self.eval_model.device))
        
        actor_loss = torch.cat(actor_loss)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.Actor.optimizer.step()
        
        critic_loss = torch.cat(critic_loss)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        self.Critic.optimizer.step()
        self.decrement_epsilon()
            
        return sum(G), len(G)
        
    def play_episode(self, env):
    
        done   = False
        i      = 0
        state  = env.reset()
        
        rewards   = []
        states    = []
        actions   = []
        while (not done) or (i<self.max_iterations):
            
            action,s_pred = self.choose_action(state)
            n_state, reward, done, info = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            
            state = n_state
            
            i +=1
            
        R = 0
        # G is a vector contains the ground truth return G_t(s,a) during the trajectory s,a,r, s_n, ............
        G = rewards
        for i in range(len(rewards)-1, -1, -1):
            R = (rewards[i] + self.gamma * R)
            G[i] = R
            
        #Normalize RETURN of the states 
        G_mean = np.mean(new_rewards)
        G_std  = np.std(new_rewards)
        G      = (G-G_mean)/G_std
        
        return states, actions, G
    
    
    
    
    
    
