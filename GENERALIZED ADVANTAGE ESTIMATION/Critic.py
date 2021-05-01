# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:14:22 2020

@author: Abdelhamid 
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class Critic(nn.Module):
    def __init__(self, lr, n_actions, input_dim):
        super(PolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.loss      = nn.MSELoss()
        
    def forward(self,state):
        
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x
