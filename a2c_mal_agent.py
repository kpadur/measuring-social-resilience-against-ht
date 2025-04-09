import torch
import numpy as np
import random

from nns import Attacker1_nn, Attacker2_nn, Attacker3_nn

class A2CMalAgent(torch.nn.Module):
    def __init__(self, state_size, n_stage_actions, n_cyber_actions, n_disinfo_actions, 
                 mal_contacts_dict, malagents_dict, malagents, neighbours, attack_target, use_attack_strategy,
                 ann1_lr, ann2_lr, ann3_lr, warmup_time, attack_campaign_duration, device):
        """ Malicious agents' behaviour in the environment """
        super().__init__()
        self.device = device

        self.stage_action_nn = self.init_stage_action_nn(state_size, n_stage_actions)
        self.cyber_action_nn = self.init_cyber_action_nn(state_size, n_cyber_actions)
        self.disinfo_action_nn = self.init_disinfo_action_nn(state_size, n_disinfo_actions)
        
        self.stage_action_opt = torch.optim.Adam(self.stage_action_nn.parameters(), lr=ann1_lr)
        self.cyber_action_opt = torch.optim.Adam(self.cyber_action_nn.parameters(), lr=ann2_lr)
        self.disinfo_action_opt = torch.optim.Adam(self.disinfo_action_nn.parameters(), lr=ann3_lr)

        # # Initialise attack target and opinion which is spread
        self.n_cyber_actions = n_cyber_actions
        self.n_disinfo_actions = n_disinfo_actions
        self.mal_contacts_dict = mal_contacts_dict
        self.malagents_dict = malagents_dict
        self.attack_target = attack_target
        self.malagents = malagents
        self.neighbours = neighbours
        self.use_attack_strategy = use_attack_strategy
        self.warmup_time = warmup_time
        self.attack_campaign_duration = attack_campaign_duration

    def init_stage_action_nn(self, observation_space, n_actions):
        # Initialise attack stage action neural network here
        return Attacker1_nn(observation_space, n_actions).to(self.device)

    def init_cyber_action_nn(self, observation_space, n_actions):
        # Initialise cyberattack neural network here
        return Attacker2_nn(observation_space, n_actions).to(self.device)
    
    def init_disinfo_action_nn(self, observation_space, n_actions):
        # Initialise disinformation neural network here
        return Attacker3_nn(observation_space, n_actions).to(self.device)
        
    def sample_actions(self, states, timestep):
        """
        Samples the action to be taken in the current attack stage
        state:          numpy array of shape (timestep, state_shape),
                        shows the environment states
        attack_stage:   int,
                        shows the current attack stage
        return:         int,
                        shows the action to be taken in the current attack stage
        """
        # Make strategical decision based on current state (time)
        if timestep > self.warmup_time and timestep <= self.warmup_time+self.attack_campaign_duration:
            action = self.use_attack_strategy[timestep]
        elif timestep <= self.warmup_time:
            action = 0
        else:
            action = 7

        # Make tactical decisions based on the attack stage
        if action in [0, 7]:
            bot_action = -1; sn_actions_dict = {f"malagent{magent}": -1 for magent in self.malagents}; botnet = {}
        elif action in [1, 4]: # start cyberattack, continue cyberattack
            bot_action, botnet = self.sample_bot_size(states[-1])
            sn_actions_dict = {f"malagent{magent}": -1 for magent in self.malagents}
        elif action in [2, 5]: # start disinfo, continue disinfo
            bot_action = -1; botnet = {}
            sn_actions_dict = self.sample_contacts(states[-1])
        else:
            bot_action, botnet = self.sample_bot_size(states[-1])
            sn_actions_dict = self.sample_contacts(states[-1])

        return action, bot_action, sn_actions_dict, botnet
    
    def sample_bot_size(self, state):
        """
        Samples bot size as the amount of resources to be used in the attack
        state:  numpy array of shape (state_shape),
        action: int,
                shows the action to be taken in the current attack stage
        return: int, int, list,
                shows attack target id, bot size, and list of attackers in botnet
        """
        # Detemine the target and bot size based on the selected action in current attack stage

        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.cyber_action_nn(state)
        action_probs = torch.nn.functional.softmax(logits, -1).cpu().detach().numpy().squeeze(0)
        # Determine the amount of resources to use to conduct an attack (can be 0 that means 1, etc.)
        bot_action = np.random.choice(self.n_cyber_actions, p = action_probs)
        bot_size = bot_action + 1
        # Assign the number of attackers to bot using sampled bot size
        botnet_ids = np.random.choice(self.malagents, size=bot_size, replace = False) # list of attackers in botnet
        # Form botnet as a dictionary of {attacker_id: target}
        botnet = {f"malagent{id}": self.attack_target for id in botnet_ids}

        return bot_action, botnet

    def sample_contacts(self, state):
        """
        Samples the contacts to be made in the current attack stage;
        each malicious agent is assigned a contact to spread disinformation to.
        state:  numpy array of shape (state_shape),
        action: int,
                shows the action to be taken in the current attack stage
        return: int, numpy array of shape (n_malagents),
                shows opinion to be spread, and array of whom each malicious agent contacts
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits,_ = self.disinfo_action_nn(state)
        action_probs = torch.nn.functional.softmax(logits, -1).cpu().detach().numpy().squeeze(0)

        # Convert dictionary to a list of contacts and corresponding actions
        contact_action_pairs = list(self.mal_contacts_dict.items())
        # Shuffle contact-action pairs to ensure random allocation if probabilities are equal
        random.shuffle(contact_action_pairs)
        # Sort contact-action pairs by action probability, in descending order
        contact_action_pairs.sort(key=lambda pair: action_probs[pair[1]], reverse=True)
        # Create a dictionary of {malagent: contact} for the current timestep
        sn_actions_dict = {f"malagent{magent}": -1 for magent in self.malagents}
        
        for i, magent in enumerate(self.malagents):
            # Get list of contacts for the specific malicious agent
            contacts = self.neighbours[magent]
            # Find the highest-probability action which corresponding contact id 
            # is connected to the specific malicous agent
            for contact, action in contact_action_pairs:
                if contact in contacts:
                    # Save action (that corresponds to the contact)
                    sn_actions_dict[f"malagent{magent}"] = action
                    # Remove the chosen contact-action pair to ensure uniqueness
                    contact_action_pairs.remove((contact, action))
                    break
        return sn_actions_dict
    
    def compute_a2c_loss(self, all_states, all_actions, total_rewards, gamma_ann1, gamma_ann2, gamma_ann3, entropy_coef):
        # Extract information from dictionaries to initialise numpy arrays
        all_states = np.array(all_states)
        # Create numpy arrays from lists
        attack_stage_actions = np.array([t[0] for t in all_actions])
        attack_stage_rewards = np.array([r[0] for r in total_rewards])
        botnet_actions = np.array([t[1] for t in all_actions])
        cyberattack_rewards = np.array([r[1] for r in total_rewards])
        disinfo_dict = np.array([a[2] for a in all_actions])
        rewards_dict = np.array([r[2] for r in total_rewards])
        disinfo_actions = np.zeros((len(attack_stage_actions), len(self.malagents)), dtype = int)
        disinfo_rewards = np.zeros((len(attack_stage_actions), len(self.malagents)), dtype = float)
        for i, action_dict in enumerate(disinfo_dict):
            for malagent_name, action in action_dict.items():
                # Extract the malagent number from the name
                malagent_num = int(malagent_name.replace('malagent', ''))
                # Get the index for this malagent
                malagent_index = self.malagents_dict[malagent_num]
                # Assign the action to the correct place in disinfo_actions
                disinfo_actions[i, malagent_index] = action
        for i, reward_dict in enumerate(rewards_dict):
            for action, reward in reward_dict.items():
                # Find which malagent did this action
                malagent_indices = np.where(disinfo_actions[i] == action)
                # It's possible that the same action is performed by multiple agents,
                # so we need to assign the reward to all of them.
                for malagent_index in malagent_indices:
                    disinfo_rewards[i, malagent_index] = reward
        # Start loss calculation
        # Get rewards to go
        n_timesteps = attack_stage_rewards.shape[0] # total number of individual rewards
        rewards_to_go = np.zeros_like(attack_stage_rewards) # initialise empty array to return the rewards to go
        rewards_to_go[-1] = attack_stage_rewards[-1]
        
        for t in range(n_timesteps-2, -1, -1): # go from timestep-2 to 0 (backwards in time)
            rewards_to_go[t] = attack_stage_rewards[t] + gamma_ann1 * rewards_to_go[t+1]
 
        # Convert numpy array to torch tensors
        states = torch.tensor(all_states, device=self.device, dtype=torch.float32).unsqueeze(0) # add batch dimension
        attack_stage_actions = torch.tensor(attack_stage_actions, device=self.device, dtype=torch.long).unsqueeze(0) # add batch dimension
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float) 
        # Get action probabilities from states
        logits, state_values = self.stage_action_nn(states)

        probs = torch.nn.functional.softmax(logits, -1) 
        log_probs = torch.nn.functional.log_softmax(logits, -1)

        # Select the log probabilities corresponding to the chosen actions
        log_probs_for_actions = log_probs[range(len(attack_stage_actions)), attack_stage_actions] 

        # Calculate advantage A(s_t,a_t) (MC discounted version)
        advantage = rewards_to_go - state_values.squeeze(-1)

        # Compute loss to be minimised
        J = torch.mean(log_probs_for_actions * advantage)
        H = -(probs * log_probs).sum(-1).mean() # entropy of a distribution
        loss3 = -(J + entropy_coef * H)
        # loss3 = Variable(loss3, requires_grad=True)

        # Backpropagate loss
        opt1 = self.stage_action_opt
        opt1.zero_grad()
        loss3.backward()
        opt1.step()

        # Calculate losses for cyberattack and disinformation
        loss4 = self.compute_loss_attacker_cyber_actions(all_states, botnet_actions, cyberattack_rewards, gamma_ann2, entropy_coef)
        loss5 = self.compute_loss_attacker_disinfo_actions(all_states, disinfo_actions, disinfo_rewards, gamma_ann3, entropy_coef)

        return loss3, loss4, loss5
    
    def compute_loss_attacker_cyber_actions(self, states, botnet_actions, cyberattack_rewards, gamma_ann2, entropy_coef):
        # Get rewards to go
        n_timesteps = cyberattack_rewards.shape[0] # total number of individual rewards
        rewards_to_go = np.zeros_like(cyberattack_rewards) # initialise empty array to return the rewards to go
        rewards_to_go[-1] = cyberattack_rewards[-1]
        
        for t in range(n_timesteps-2, -1, -1): # go from timestep-2 to 0 (backwards in time)
            rewards_to_go[t] = cyberattack_rewards[t] + gamma_ann2 * rewards_to_go[t+1]
        # Convert numpy array to torch tensors
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        botnet_actions = torch.tensor(botnet_actions, device=self.device, dtype=torch.long)
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)
        
        # Get action probabilities from states
        logits, state_values = self.cyber_action_nn(states) # state values V(s_t)
        probs = torch.nn.functional.softmax(logits, -1)
        log_probs = torch.nn.functional.log_softmax(logits, -1).squeeze(0)

        # Select the log probabilities corresponding to the chosen actions
        log_probs_for_actions = log_probs[range(len(botnet_actions)), botnet_actions]

        # Calculate advantage A(s_t,a_t) (MC discounted version) to calculate the extra reward we get
        # if we take this action at that state compared to the mean reward we get at that state.
        advantage = rewards_to_go - state_values.squeeze(-1) 

        # Compute loss to be minimised
        J = torch.mean(log_probs_for_actions*(advantage)) # J = \frac{1}{T} \sum_{t=1}^T log\pi_\theta(a_t|s_t)\hat{A}(s_t,a_t)
        H = -(probs*log_probs).sum(-1).mean() # H = -\frac{1}{T} \sum_{t=1}^T \sum_a \pi_\theta(a_t|s_t)log\pi_\theta(a_t|s_t)
        # Loss = -\frac{1}{T}\sum_{t=1}^T(log pi_\theta(a_t|s_t)A(s_t,a_t) - \beta H(\pi_\theta))
        loss4 = -(J+entropy_coef*H)
        # loss4 = Variable(loss4, requires_grad=True)

        # Backpropagate loss
        self.cyber_action_opt.zero_grad()
        loss4.backward()
        self.cyber_action_opt.step()

        return loss4

    def compute_loss_attacker_disinfo_actions(self, states, init_disinfo_actions, init_disinfo_rewards, gamma_ann3, entropy_coef):
        # Initialize binary actions and rewards arrays
        disinfo_actions = np.zeros((init_disinfo_actions.shape[0], self.n_disinfo_actions), dtype=int)
        disinfo_rewards = np.zeros((init_disinfo_actions.shape[0], self.n_disinfo_actions), dtype=int)

        # Assign binary actions and accumulate rewards
        for episode, actions in enumerate(init_disinfo_actions):
            for action_index, action in enumerate(actions):
                if action != -1:
                    disinfo_actions[episode, action] = 1
                    # Accumulate rewards
                    disinfo_rewards[episode, action] = init_disinfo_rewards[episode, action_index]

        # Calculate rewards to go
        n_timesteps, n_actions = disinfo_rewards.shape
        rewards_to_go = np.zeros_like(disinfo_rewards)
        rewards_to_go[-1] = disinfo_rewards[-1]

        for t in range(n_timesteps - 2, -1, -1):
            for a in range(n_actions):
                rewards_to_go[t, a] = gamma_ann3 * rewards_to_go[t + 1, a] + disinfo_rewards[t, a]
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        disinfo_actions = torch.tensor(disinfo_actions, device=self.device, dtype=torch.long)
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)

        logits, state_values = self.disinfo_action_nn(states)
        probs = torch.sigmoid(logits) # using sigmoid function
        log_probs = torch.log(probs) # shape ([timestep, agents]) # has to be changed
        log_probs_for_actions = log_probs*disinfo_actions + (1-log_probs)*(1-disinfo_actions) # used for classification

        # Calculate advantages for each agent and then average them
        advantage_f = rewards_to_go - state_values.unsqueeze(1)

        J = (log_probs_for_actions*advantage_f.detach()).mean(1).mean()
        H = -(probs*log_probs_for_actions).sum(-1).mean()
        loss = -(J + entropy_coef*H)

        # Backpropagate loss
        self.disinfo_action_opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.disinfo_action_nn.parameters(), max_norm=0.9)
        self.disinfo_action_opt.step()


        return loss