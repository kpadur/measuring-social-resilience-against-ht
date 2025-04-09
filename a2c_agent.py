import torch
import numpy as np

from nns import Action_nn, Opinion_nn

class A2CRegAgent:
    def __init__(self, state_shape, n_actions, n_opinions, rnn1_lr, rnn2_lr, device):
        self.device = device

        # Initialise action and opinion networks
        self.action_nn = self.init_action_nn(state_shape, n_actions)
        self.opinion_nn = self.init_opinion_nn(state_shape, n_opinions)

        # Initialise optimisers for the networks
        self.action_opt = torch.optim.Adam(self.action_nn.parameters(), lr=rnn1_lr)
        self.opinion_opt = torch.optim.Adam(self.opinion_nn.parameters(), lr=rnn2_lr)

    def init_action_nn(self, state_shape, action_space):
        # Initialise action neural network
        return Action_nn(state_shape, action_space).to(self.device)

    def init_opinion_nn(self, state_shape, action_space):
        # Initialise opinion neural network
        return Opinion_nn(state_shape, action_space).to(self.device)

    def sample_actions(self, state):
        """
        Sample actions and opinions for a specific agent.
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)

        # Sample action
        with torch.no_grad():
            action_logits, _ = self.action_nn(state)
        action_probs = torch.nn.functional.softmax(action_logits, -1).cpu().detach().numpy().squeeze(0)
        action = np.random.choice(len(action_probs), p=action_probs)

        # Sample opinion
        with torch.no_grad():
            opinion_logits, _ = self.opinion_nn(state)
        opinion_probs = torch.nn.functional.softmax(opinion_logits, -1).cpu().detach().numpy().squeeze(0)
        opinion = np.random.choice(len(opinion_probs), p=opinion_probs)

        return action, opinion

    def compute_a2c_loss(self, states, all_actions, total_rewards, gamma_rnn1, gamma_rnn2, entropy_coef):
        """
        Compute the losses for actions and opinions.
        """
        states = np.array(states)
        actions = np.array([t[0] for t in all_actions])
        opinions = np.array([t[1] for t in all_actions])
        rewards = np.array([r[0] for r in total_rewards])
        feedback = np.array([r[1] for r in total_rewards])

        rewards_to_go = self.compute_rewards_to_go(rewards, gamma_rnn1)
        feedback_to_go = self.compute_rewards_to_go(feedback, gamma_rnn2)

        # Convert to tensors
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)
        feedback_to_go = torch.tensor(feedback_to_go, device=self.device, dtype=torch.float)
        states_t = torch.tensor(states, device=self.device, dtype=torch.float)
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.long)
        opinions_t = torch.tensor(opinions, device=self.device, dtype=torch.long)

        # Action loss computation
        logits_a, state_values_a = self.action_nn(states_t)
        loss_a = self.compute_policy_loss(logits_a, state_values_a, rewards_to_go, actions_t, entropy_coef)

        # Opinion loss computation
        logits_o, state_values_o = self.opinion_nn(states_t)
        loss_o = self.compute_policy_loss(logits_o, state_values_o, feedback_to_go, opinions_t, entropy_coef)

        # Update action network
        self.action_opt.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.action_nn.parameters(), max_norm=1.0)
        self.action_opt.step()

        # Update opinion network
        self.opinion_opt.zero_grad()
        loss_o.backward()
        torch.nn.utils.clip_grad_norm_(self.opinion_nn.parameters(), max_norm=1.0)
        self.opinion_opt.step()

        return loss_a, loss_o

    @staticmethod
    def compute_rewards_to_go(rewards, gamma):
        n_timesteps = rewards.shape[0]
        rewards_to_go = np.zeros_like(rewards)
        rewards_to_go[-1] = rewards[-1]
        for t in range(n_timesteps - 2, -1, -1):
            rewards_to_go[t] = rewards[t] + gamma * rewards_to_go[t + 1]
        return rewards_to_go

    @staticmethod
    def compute_policy_loss(logits, state_values, rewards_to_go, actions, entropy_coef):
        probs = torch.nn.functional.softmax(logits, -1)
        log_probs = torch.nn.functional.log_softmax(logits, -1)
        log_probs_for_actions = log_probs[range(len(actions)), actions]
        advantage = rewards_to_go - state_values.squeeze(-1)
        J = torch.mean(log_probs_for_actions * advantage)
        H = -(probs * log_probs).sum(-1).mean()
        return -(J + entropy_coef * H)
