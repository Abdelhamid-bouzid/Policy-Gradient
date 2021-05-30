import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.n_actions  = n_actions
        self.fc1        = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2        = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi         = nn.Linear(self.fc2_dims, n_actions)
        self.v          = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer  = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state):
        x  = F.relu(self.fc1(state))
        x  = F.relu(self.fc2(x))
        pi = self.pi(x)
        v  = self.v(x)
        return (pi, v)