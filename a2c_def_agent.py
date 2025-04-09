import torch
import numpy as np

from nns import Defender1_nn, Defender2_nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu

class A2CServiceProvider:
    """Service providers' behaviour"""

    def __init__(self, state_space_defenders, n_filter, n_answer, dnn1_lr, dnn2_lr, \
                 warmup_time, attack_campaign_duration, device):
        super().__init__()
        self.device = device

        self.warmup_time = warmup_time
        self.attack_campaign_duration = attack_campaign_duration
        self.n_filter = n_filter
        self.n_answer = n_answer

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

    def sample_actions(self, state, timestep):
        """
        Filters out the malicious traffic from the legitimate traffic
        at this time step (Response)
        defender:   defender ID
        state:      numpy binary array of shape (n_agents),
                    represemts whether each agent made a request (1) or not (0)
        return:     numpy binary array of shape (n_agents),
                    the final decision on each agent's request after applying
                    the actions decided by the defender
        """
        if timestep > self.warmup_time and timestep <= self.warmup_time + self.attack_campaign_duration:
            req_state = torch.tensor(state[0], device=device, dtype=torch.float32).unsqueeze(0)

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

            sn_state = torch.tensor(state[1], device=device, dtype=torch.float32).unsqueeze(0) # should be state 1 and 2 
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
        else:
            actions = np.zeros(self.n_filter)
            sn_actions = np.zeros(self.n_answer)
        
        return actions, sn_actions

    def compute_a2c_loss(self, states, all_actions, total_rewards, gamma_dnn1, gamma_dnn2, entropy_coef):

        state_requests = np.array([s[0] for s in states])
        state_sn = np.array([s[1] for s in states])
        filters = np.array([t[0] for t in all_actions])
        answers = np.array([t[1] for t in all_actions])
        rewards = np.array([r[0] for r in total_rewards])
        feedback = np.array([r[1] for r in total_rewards])

        n_timesteps, n_actions = rewards.shape
        rewards_to_go = np.zeros_like(rewards)
        rewards_to_go[-1] = rewards[-1]

        for t in range(n_timesteps - 2, -1, -1):
            for a in range(n_actions):
                rewards_to_go[t, a] = gamma_dnn1 * rewards_to_go[t + 1, a] + rewards[t, a]
        

        n_timesteps, n_actions = rewards.shape
        feedback_to_go = np.zeros_like(feedback)
        feedback_to_go[-1] = feedback[-1]

        for t in range(n_timesteps - 2, -1, -1):
            for a in range(n_actions):
                feedback_to_go[t, a] = gamma_dnn2 * feedback_to_go[t + 1, a] + feedback[t, a]

        # Convert to tensors
        state_requests = torch.tensor(state_requests, device=device, dtype=torch.float)
        state_sn = torch.tensor(state_sn, device=device, dtype=torch.float)
        filters = torch.tensor(filters, device=device, dtype=torch.long)
        answers = torch.tensor(answers, device=device, dtype=torch.long)
        rewards_to_go = torch.tensor(rewards_to_go, device=device, dtype=torch.float)
        feedback_to_go = torch.tensor(feedback_to_go, device=device, dtype=torch.float)

        logits_f, state_values_f = self.filter_nn(state_requests)
        probs_f = torch.sigmoid(logits_f) # using sigmoid function
        log_probs_f = torch.log(probs_f) # shape ([timestep, agents]) # has to be changed
        log_probs_for_filters = log_probs_f*filters + (1-log_probs_f)*(1-filters) # used for classification

        # Calculate advantages for each agent and then average them
        advantage_f = rewards_to_go - state_values_f.unsqueeze(1)

        J_f = (log_probs_for_filters*advantage_f.detach()).mean(1).mean()
        H_f = -(probs_f*log_probs_for_filters).sum(-1).mean()
        loss_f = -(J_f + entropy_coef*H_f)

        # Backpropagate loss
        self.filter_opt.zero_grad()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(self.filter_nn.parameters(), max_norm=0.5)
        self.filter_opt.step()

        logits_a, state_values_a = self.answer_nn(state_sn)
        probs_a = torch.sigmoid(logits_a) # using sigmoid function
        log_probs_a = torch.log(probs_a) # shape ([timestep, agents]) # has to be changed
        log_probs_for_answers = log_probs_a*filters + (1-log_probs_a)*(1-answers) # used for classification

        # Calculate advantages for each agent and then average them
        advantage_a = feedback_to_go - state_values_a.unsqueeze(1)

        J_a = (log_probs_for_answers*advantage_a.detach()).mean(1).mean()
        H_a = -(probs_a*log_probs_for_answers).sum(-1).mean()
        loss_a = -(J_a + entropy_coef*H_a)

        # Backpropagate loss
        self.answer_opt.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.answer_nn.parameters(), max_norm=0.5)
        self.answer_opt.step()

        return loss_f, loss_a