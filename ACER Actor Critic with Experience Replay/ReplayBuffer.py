import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size         = max_size
        self.mem_cntr         = 0
        self.state_memory     = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.actions          = np.zeros(self.mem_size, dtype=np.int)
        self.log_probs        = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, log_prob, reward, state_, done):
        index                       = self.mem_cntr % self.mem_size
        
        self.state_memory[index]     = state
        self.new_state_memory[index] = state_
        self.actions[index]          = action
        self.log_probs[index]        = log_prob
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done
        
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch   = np.random.choice(max_mem, batch_size, replace=False)

        states   = self.state_memory[batch]
        actions  = self.actions[batch]
        probs    = self.log_probs[batch]
        rewards  = self.reward_memory[batch]
        states_  = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, probs, rewards, states_, terminal
