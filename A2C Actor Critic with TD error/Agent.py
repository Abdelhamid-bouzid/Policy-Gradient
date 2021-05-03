import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    def __init__(self, lr, input_dims, n_actions, gamma=0.99,l1_size=256, l2_size=256, batch_size=32, mem_size=1000000):
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.memory       = ReplayBuffer(mem_size, input_dims)
        self.actor_critic = ActorCriticNetwork(lr, input_dims, l1_size,l2_size, n_actions=n_actions)
        self.log_probs    = []

    def store_transition(self, state, log_prob, reward, state_, done):
        self.memory.store_transition(state, log_prob, reward, state_, done)

    def choose_action(self, observation):
        state            = T.tensor([observation]).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities    = F.softmax(probabilities)
        action_probs     = T.distributions.Categorical(probabilities)
        action           = action_probs.sample()
        log_probs        = action_probs.log_prob(action)

        return action.item(), log_probs

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.actor_critic.optimizer.zero_grad()

        state, prob, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states  = T.tensor(state).to(self.actor_critic.device)
        probs   = T.tensor(prob).to(self.actor_critic.device)
        rewards = T.tensor(reward).to(self.actor_critic.device)
        dones   = T.tensor(done).to(self.actor_critic.device)
        states_ = T.tensor(new_state).to(self.actor_critic.device)

        _, critic_value_ = self.actor_critic.forward(states_)
        _, critic_value  = self.actor_critic.forward(states)

        critic_value_[dones] = 0.0

        delta = rewards + self.gamma*critic_value_ - critic_value

        actor_loss  = -T.mean(probs*delta)
        critic_loss = F.mse_loss(rewards + self.gamma*critic_value_, critic_value)

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()