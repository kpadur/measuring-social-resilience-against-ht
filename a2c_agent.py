import torch
import numpy as np

from nns import Action_nn, Opinion_nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu

class A2CRegAgent:
    def __init__(self, state_shape, n_actions, n_opinions, rnn1_lr, rnn2_lr, device):
        self.device = device

        self.action_nn = self.init_action_nn(state_shape, n_actions)
        self.opinion_nn = self.init_opinion_nn(state_shape, n_opinions)
        
        self.action_opt = torch.optim.Adam(self.action_nn.parameters(), lr=rnn1_lr)
        self.opinion_opt = torch.optim.Adam(self.opinion_nn.parameters(), lr=rnn2_lr)

    def init_action_nn(self, state_shape, action_space):
        # Initialize action neural network here
        # return Action_nn(len(observation_space), action_space[0].n).to(self.device)
        return Action_nn(state_shape, action_space).to(self.device)

    def init_opinion_nn(self, state_shape, action_space):
        # Initialize opinion neural network here
        # return Opinion_nn(len(observation_space), action_space[1].n).to(self.device)
        return Opinion_nn(state_shape, action_space).to(self.device)

    def sample_actions(self, state):
        # Sample actions from the action neural network
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = self.action_nn(state)
        action_probs = torch.nn.functional.softmax(action_logits, -1).cpu().detach().numpy().squeeze(0)
        action = np.random.choice(len(action_probs), p=action_probs)

        # Sample opinions from the opinion neural network
        with torch.no_grad():
            opinion_logits, _ = self.opinion_nn(state)
        opinion_probs = torch.nn.functional.softmax(opinion_logits, -1).cpu().detach().numpy().squeeze(0)
        opinion = np.random.choice(len(opinion_probs), p=opinion_probs)

        return action, opinion
    
    def compute_a2c_loss(self, states, all_actions, total_rewards, gamma_rnn1, gamma_rnn2, entropy_coef):
        """
        Compute the loss for training the agent using actions and rewards-to-go.
        The loss combines the policy gradient objective(J) and the entropy regularisation term (H),
        and it is used to update the model parameters via backpropagation.
        """
        actions = np.array([t[0] for t in all_actions])
        opinions = np.array([t[1] for t in all_actions])
        rewards = np.array([r[0] for r in total_rewards])
        feedback = np.array([r[1] for r in total_rewards])
        states = np.array(states)

        # Rewards to go for actions
        n_timesteps = rewards.shape[0] # total number of individual rewards
        rewards_to_go = np.zeros_like(rewards) # initialise empty array to return the rewards to go
        rewards_to_go[-1] = rewards[-1]
        
        for t in range(n_timesteps-2, -1, -1): # go from timestep-2 to 0 (backwards in time)
            rewards_to_go[t] = rewards[t] + gamma_rnn1 * rewards_to_go[t+1]

        # Rewards to go for opinions
        n_timesteps = feedback.shape[0]
        feedback_to_go = np.zeros_like(feedback)
        feedback_to_go[-1] = feedback[-1]

        for t in range(n_timesteps-2, -1, -1):
            feedback_to_go[t] = feedback[t] + gamma_rnn2 * feedback_to_go[t+1]
        
        # Convert to tensors
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)
        feedback_to_go = torch.tensor(feedback_to_go, device=self.device, dtype=torch.float)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        opinions = torch.tensor(opinions, device=self.device, dtype=torch.long)
        
        # Deal with actions
        logits_a, state_values_a = self.action_nn(states)
        probs_a = torch.nn.functional.softmax(logits_a, -1) 

        log_probs_a = torch.nn.functional.log_softmax(logits_a, -1)
        log_probs_for_actions = log_probs_a[range(len(actions)), actions] 
        advantage_a = rewards_to_go - state_values_a.squeeze(-1)

        J_a = torch.mean(log_probs_for_actions*(advantage_a))
        H_a = -(probs_a*log_probs_a).sum(-1).mean()
        loss_a = -(J_a+entropy_coef*H_a)

        # Backpropagate loss
        self.action_opt.zero_grad() # reset gradients
        loss_a.backward() # backpropagate
        self.action_opt.step() # update parameters using gradient ascent

        # Deal with opinions
        logits_o, state_values_o = self.opinion_nn(states)
        probs_o = torch.nn.functional.softmax(logits_o, -1)

        log_probs_o = torch.nn.functional.log_softmax(logits_o, -1)
        log_probs_for_opinions = log_probs_o[range(len(opinions)), opinions]

        advantage_o = feedback_to_go - state_values_o.squeeze(-1)

        J_o = torch.mean(log_probs_for_opinions*(advantage_o))
        H_o = -(probs_o*log_probs_o).sum(-1).mean()
        loss_o = -(J_o+entropy_coef*H_o)

        # Backpropagate loss
        self.opinion_opt.zero_grad()
        loss_o.backward()
        self.opinion_opt.step()
        
        return loss_a, loss_o
