import torch
import numpy as np

from nns import Defender1_nn, Defender2_nn

class A2CServiceProvider:
    """Service provider's behaviour"""

    def __init__(self, state_space_defenders, n_filter, n_answer, dnn1_lr, dnn2_lr, device):
        super().__init__()
        self.device = device

        self.filter_nn = self.init_filter_nn(state_space_defenders, n_filter)
        self.answer_nn = self.init_answer_nn(state_space_defenders, n_answer)
        self.filter_opt = torch.optim.Adam(self.filter_nn.parameters(), lr=dnn1_lr)
        self.answer_opt = torch.optim.Adam(self.answer_nn.parameters(), lr=dnn2_lr)

    def init_filter_nn(self, observation_space, n_actions):
        # Initialise action neural network here
        return Defender1_nn(observation_space[1], n_actions).to(self.device)

    def init_answer_nn(self, observation_space, n_actions):
        # Initialize opinion neural network here
        return Defender2_nn(observation_space[1], n_actions).to(self.device)

    def sample_actions(self, state):
        """
        Filters out the malicious traffic from the legitimate traffic
        and answers to agents in the social network at this time step.
        """
        req_state = torch.tensor(state[0], device=self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Get the probabilities of how likely the defender should block the request for each agent
            logits_filter, _ = self.filter_nn(req_state)
        # Apply sigmoid function to the logits to get probabilities
        probs_filter = torch.sigmoid(logits_filter)
        # Sample from the Bernoulli probability distribution for each agent to get the actions
        actions = torch.distributions.Bernoulli(probs_filter).sample().squeeze(0)
        # Convert tensor to a numpy array
        actions = actions.cpu().numpy().astype(int)
        # Apply 0 to agents who did not request service
        actions = actions * state[0]

        sn_state = torch.tensor(state[1], device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # Get the probabilities of how likely the defender should reply to each agent
            logits_answer, _ = self.answer_nn(sn_state)
        # Apply sigmoid function to the logits to get probabilities
        probs_answer = torch.sigmoid(logits_answer)
        # Sample from the Bernoulli probability distribution for each agent to get the actions
        sn_actions = torch.distributions.Bernoulli(probs_answer).sample().squeeze(0)
        # Convert tensor to a numpy array
        sn_actions = sn_actions.cpu().numpy().astype(int)
        # Apply actions only to agents who expressed negative opinion (only) about the service provider
        sn_actions = sn_actions * state[1]
        
        return actions, sn_actions

    def compute_a2c_loss(self, states, all_actions, total_rewards, gamma_dnn1, gamma_dnn2, entropy_coef):
        """
        Compute the losses for filtering and replying decisions.
        """
        state_requests = np.array([s[0] for s in states])
        state_sn = np.array([s[1] for s in states])
        filters = np.array([t[0] for t in all_actions])
        answers = np.array([t[1] for t in all_actions])
        rewards = np.array([r[0] for r in total_rewards])
        feedback = np.array([r[1] for r in total_rewards])

        # Calculate rewards to go
        rewards_to_go = self.compute_rewards_to_go(rewards, gamma_dnn1)
        feedback_to_go = self.compute_rewards_to_go(feedback, gamma_dnn2)

        # Convert to tensors
        state_requests = torch.tensor(state_requests, device=self.device, dtype=torch.float)
        state_sn = torch.tensor(state_sn, device=self.device, dtype=torch.float)
        filters = torch.tensor(filters, device=self.device, dtype=torch.long)
        answers = torch.tensor(answers, device=self.device, dtype=torch.long)
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)
        feedback_to_go = torch.tensor(feedback_to_go, device=self.device, dtype=torch.float)

        # Calculate loss
        logits_f, state_values_f = self.filter_nn(state_requests)
        loss_f = self.compute_policy_loss(logits_f, state_values_f, rewards_to_go, filters, entropy_coef)

        # Backpropagate loss
        self.filter_opt.zero_grad()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(self.filter_nn.parameters(), max_norm=0.5)
        self.filter_opt.step()

        # Calculate loss
        logits_a, state_values_a = self.answer_nn(state_sn)
        loss_a = self.compute_policy_loss(logits_a, state_values_a, feedback_to_go, answers, entropy_coef)

        # Backpropagate loss
        self.answer_opt.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.answer_nn.parameters(), max_norm=0.5)
        self.answer_opt.step()

        return loss_f, loss_a
    
    @staticmethod
    def compute_rewards_to_go(rewards, gamma):
        n_timesteps, n_actions = rewards.shape
        rewards_to_go = np.zeros_like(rewards)
        rewards_to_go[-1] = rewards[-1]

        for t in range(n_timesteps - 2, -1, -1):
            for a in range(n_actions):
                rewards_to_go[t, a] = gamma * rewards_to_go[t + 1, a] + rewards[t, a]
        return rewards_to_go

    @staticmethod
    def compute_policy_loss(logits, state_values, rewards_to_go, actions, entropy_coef):
        probs = torch.sigmoid(logits) # using sigmoid function
        log_probs = torch.log(probs)
        log_probs_for_actions = log_probs*actions + (1-log_probs)*(1-actions)
        # Calculate advantages for each agent and then average them
        advantage = rewards_to_go - state_values.unsqueeze(1)
        J = (log_probs_for_actions*advantage.detach()).mean(1).mean()
        H = -(probs*log_probs_for_actions).sum(-1).mean()
        return -(J + entropy_coef * H)
