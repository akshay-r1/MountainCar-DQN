import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Defining the Deep Q-Network
class Network(nn.Module):
    def __init__(self, input_shape, hidden1, hidden2, n_actions, rate=0.00001):

        super(Network,self).__init__()
        self.input_shape = input_shape
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.n_actions = n_actions
        self.rate = rate

        # Define the layers
        self.fc1 = nn.Linear(self.input_shape,self.hidden1)
        self.fc2 = nn.Linear(self.hidden1,self.hidden2)
        self.final = nn.Linear(self.hidden2,self.n_actions)

        # Choice of  loss and optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=self.rate)
        self.loss = nn.MSELoss()

    def forward(self, observation):
        state = torch.Tensor(observation)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.final(x)

        return actions