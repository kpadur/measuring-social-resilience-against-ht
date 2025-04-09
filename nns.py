import torch
import torch.nn as nn

class Attacker1_nn(torch.nn.Module):
    def __init__(self, state_size, n_stage_actions):
        super().__init__()
        self.out_size = n_stage_actions
        self.in_size = state_size
        self.hidden_size1 = 192
        self.hidden_size2 = 128
        self.hidden_size3 = 64
        self.n_layers = 5

        self.lstm = torch.nn.LSTM(self.in_size, self.hidden_size1, self.n_layers, batch_first=True)
        self.l2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        self.l3 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)
        self.actor = torch.nn.Linear(self.hidden_size3, self.out_size)
        self.critic = torch.nn.Linear(self.hidden_size3, 1)

    def forward(self, s):
        h0 = torch.zeros(self.n_layers, s.size(0), self.hidden_size1, device = s.device)
        c0 = torch.zeros(self.n_layers, s.size(0), self.hidden_size1, device = s.device)
        x, _ = self.lstm(s, (h0, c0))
        z = torch.tanh(self.l2(x[:,-1,:]))
        y = torch.tanh(self.l3(z))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value

class Attacker2_nn(torch.nn.Module):
    def __init__(self, state_size, n_cyber_actions):

        super().__init__()
        self.n_cyber_actions = n_cyber_actions
        
        state_dim = state_size
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 192)
        self.l3 = torch.nn.Linear(192, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.actor = torch.nn.Linear(64, self.n_cyber_actions)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, s):
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        w = torch.tanh(self.l3(z))
        y = torch.tanh(self.l4(w))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value

class Attacker3_nn(torch.nn.Module):
    def __init__(self, state_size, n_misinfo_actions):

        super().__init__()
        self.n_misinfo_actions = n_misinfo_actions
        
        state_dim = state_size
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 192)
        self.l3 = torch.nn.Linear(192, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.actor = torch.nn.Linear(64, self.n_misinfo_actions)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, s):
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        w = torch.tanh(self.l3(z))
        y = torch.tanh(self.l4(w))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value

class Defender1_nn(torch.nn.Module):
    def __init__(self, dstate_space, n_filters):
        super().__init__()

        self.n_filters = n_filters
        state_dim =  dstate_space

        self.l1 = torch.nn.Linear(state_dim, 192)
        self.l2 = torch.nn.Linear(192,128)
        self.l3 = torch.nn.Linear(128,64)
        self.actor = torch.nn.Linear(64, self.n_filters)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, s):
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        y = torch.tanh(self.l3(z))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value
    
class Defender2_nn(torch.nn.Module):
    def __init__(self, dstate_space, n_answers):
        super().__init__()

        self.n_answers = n_answers
        state_dim =  dstate_space

        self.l1 = torch.nn.Linear(state_dim, 192)
        self.l2 = torch.nn.Linear(192,128)
        self.l3 = torch.nn.Linear(128,64)
        self.actor = torch.nn.Linear(64, self.n_answers)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, s):
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        y = torch.tanh(self.l3(z))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value

class Action_nn(torch.nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.n_actions = n_actions
 
        # Neural network
        state_dim = state_shape

        # Define layers
        self.l1 = torch.nn.Linear(state_dim, 192)
        self.l2 = torch.nn.Linear(192, 128)
        self.l3 = torch.nn.Linear(128, 64)
        self.actor = torch.nn.Linear(64, self.n_actions)
        self.critic = torch.nn.Linear(64, 1)  # 1 represents state value V(s)

    def forward(self, s):
        # For actions
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        y = torch.tanh(self.l3(z))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value

class Opinion_nn(torch.nn.Module):
    def __init__(self, state_shape, n_opinions):
        super().__init__()
        self.n_opinions = n_opinions
        
        # Neural network
        state_dim = state_shape

        # Define layers
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 192)
        self.l3 = torch.nn.Linear(192, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.actor = torch.nn.Linear(64, self.n_opinions)
        self.critic = torch.nn.Linear(64, 1)  # 1 represents state value V(s)

    def forward(self, s):
        # Forward pass through the network
        x = torch.tanh(self.l1(s))
        z = torch.tanh(self.l2(x))
        w = torch.tanh(self.l3(z))
        y = torch.tanh(self.l4(w))
        logits = self.actor(y)
        state_value = self.critic(y)
        return logits, state_value
