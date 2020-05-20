from dqn import Network
import torch
import numpy as np
import random


class Agent(object):
    def __init__(self, gamma, eps, eps_decay, eps_min, batch_size, input_size,
                 n_actions, rate, hidden1=100, hidden2=100, max_mem_size=100000,
                 eps_end=0.01, eps_dec=0.9):
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_actions = n_actions
        self.network = Network(input_shape=input_size, hidden1=hidden1,
                               hidden2=hidden2, n_actions=self.n_actions, rate=rate)
        self.target = Network(input_shape=input_size, hidden1=hidden1,
                              hidden2=hidden2, n_actions=self.n_actions, rate=rate)
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.action_values = np.array(self.action_space, dtype=np.uint8)

        # Define Replay Memory objects
        self.mem_size = max_mem_size
        self.state_memory = np.zeros((self.mem_size, self.input_size))
        self.action_memory = np.zeros((self.mem_size, self.n_actions))
        self.new_state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.uint8)
        self.cost_memory = np.zeros(self.mem_size)

        self.mem_counter = 0

        # set agent to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        '''Function to select the action given a state'''
        rand = np.random.random()
        if rand < self.epsilon:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
            # action = [random.uniform(-1,1),random.uniform(-1,1)]
        else:
            action = self.network.forward(state)
            # action = action.item()
            action = torch.argmax(action)
        return action.item()

    def push(self, state, action, new_state, reward):
        '''Function to store into replay memory.'''
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.cost_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.mem_counter += 1

    def learn_from_replay(self):
        '''Learning and updating the network during each time step.'''
        if self.mem_counter < self.batch_size:
            return
        self.network.optimizer.zero_grad()
        batch = np.random.choice(min(self.mem_counter, self.mem_size), self.batch_size)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        cost_batch = self.cost_memory[batch]
        new_state_batch = self.new_state_memory[batch]

        #
        action_values = np.array(self.action_space, dtype=np.uint8)
        action_indices = np.dot(action_batch, action_values)

        Q_Values = self.network.forward(state_batch)
        Q_Target = self.target.forward(state_batch)
        Q_Next = self.network.forward(new_state_batch)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        Q_Target[batch_index, action_indices] = torch.Tensor(cost_batch) + self.gamma * torch.min(Q_Next, dim=1)[0]

        loss = self.network.loss(Q_Target, Q_Values)
        loss.backward()
        self.network.optimizer.step()

        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
