# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:30:27 2021

@author: Admin
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,n_actions, name, chkpt_dir='models'):
        super(ActorNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.n_actions  = n_actions
        

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu  = nn.Linear(self.fc2_dims, self.n_actions)
        
        
        self.name            = name
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))