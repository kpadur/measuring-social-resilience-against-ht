# %% [markdown]
# ## Measuring Social Resilience against Hybrid Threats
# Chapter 5\ 
# Experiment 1\
# With this experiment, we analyse how the social component of the Cyber-Physical-Social system can plan, absorb, recover from and adapt to hybrid threats.\
# We analyse how different groups of individuals (directly affected or indirectly affected) experience disruption and how they recover from it.

# %% [markdown]
# Import libraries
from environment_exp1 import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_states_groups, process_regagent_actions, \
    process_regagent_rewards, process_service_provider_availability
from save_data import read_csv_to_dict, save_data_to_csv
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import datetime
import matplotlib.cm as cm

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 5
experiment = 1 # baseline experiment

# %% [markdown]
# Specify output directory
save_path = os.path.join("results", "exp1-results")
get_nn_path = os.path.join("regagent-parameters")

# %% [markdown]
# Specify number of agents in the environment
nProviders = 3
nRegAgents = 110
nMalAgents = 0

# %% [markdown]
# Initialise (tuned) hyperparameters
hyperparameters = read_csv_to_dict("parameters/hyperparameters.csv")
alpha_rnn1 = hyperparameters['alpha_1']
alpha_rnn2 = hyperparameters['alpha_2']
gamma_rnn1 = hyperparameters['gamma_1']
gamma_rnn2 = hyperparameters['gamma_2']
beta_start = 1
beta_end = hyperparameters['beta_1']
beta_decay = hyperparameters['n_1']

# %% [markdown]
# Initialise social network, cyber-physical system, and agent parameters
# Load parameters
parameters = read_csv_to_dict("parameters/parameters.csv")

# Social network parameters
kappa = parameters['kappa']
rho = parameters['rho']

# Cyber-physical system parameters
center_up_to_down = [parameters['center_up_to_down']] * nProviders  # Psi: prob (1 to -1)
center_down_to_up = [parameters['center_down_to_up']] * nProviders  # psi: prob (-1 to 1)
end_up_to_down = [parameters['end_up_to_down']] * nProviders        # Lambda: prob (1 to -1)
end_down_to_up = [parameters['end_down_to_up']] * nProviders        # lambda: prob (-1 to 1)
cost = [parameters['cost']] * nProviders

# Agents' attributes (parameters)
direct_exp_weight = parameters['direct_exp_weight']
feedback_adj_rate = parameters['feedback_adj_rate']
forgetting_factor = parameters['forgetting_factor']

# %% [markdown]
# Define training time, visualisation and saving frequency
n_steps = 500 # number of steps per episode
number_of_episodes = 100
vis_freq = 10
saving_freq = 10
save_fig = False

# %% [markdown]
# Initialise seed for reproducibility
seed = np.random.randint(0, 10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# Create the environment
env = Environment(nRegAgents, nProviders, 
                  kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                  direct_exp_weight, feedback_adj_rate, forgetting_factor)

# Create lists of agents, form social network of agents, and create attributes
regagents, providers, neighbours = env.regagents, env.providers, env.neighbours

# Predefine malicious agents set
malagents = [54, 30, 103, 109, 47, 76, 77, 99, 9, 69]

# Define regular agents' state space, actions, and opinions
observation_space, action_space = env.observation_spaces[f"regagent{0}"], env.action_spaces[f"regagent{0}"]

state_shape, n_actions, n_opinions = \
    env.observation_spaces[f"regagent{0}"].shape[0], env.action_spaces[f"regagent{0}"][0].n, env.action_spaces[f"regagent{0}"][1].n

# Reset environment state
envstate, info = env.reset(seed=seed)

print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)
# %% [markdown]
#  Initialise agents
regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, 
                                                  alpha_rnn1, alpha_rnn2, device) for agent in env.regagents}

for agent_name, agent in regular_agents.items():
    # Load Action NN and its optimizer
    action_checkpoint = torch.load(os.path.join(get_nn_path, f'{agent_name}_checkpoint_actions.pth'))
    agent.action_nn.load_state_dict(action_checkpoint['actions_state_dict'])
    agent.action_opt.load_state_dict(action_checkpoint['actions_opt_state_dict'])

    # Load Opinion NN and its optimizer
    opinion_checkpoint = torch.load(os.path.join(get_nn_path, f'{agent_name}_checkpoint_opinions.pth'))
    agent.opinion_nn.load_state_dict(opinion_checkpoint['opinions_state_dict'])
    agent.opinion_opt.load_state_dict(opinion_checkpoint['opinions_opt_state_dict'])

# %% [markdown]
# Collect data
total_rewards, action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1),  \
    np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
actions_history, opinions_history = np.zeros((number_of_episodes + 1, nProviders)), \
    np.zeros((number_of_episodes + 1, nProviders*2))
social_trust_history = np.zeros((number_of_episodes + 1, nProviders))

sum_sp_availability = np.zeros(nProviders, dtype=int)
sp_availability_history = np.zeros((number_of_episodes + 1, nProviders))

regagent_states_pd = pd.DataFrame()
regagent_actions_pd = pd.DataFrame()

# %% [markdown]
# Training regular agents and visualise data
for episode in range(1, number_of_episodes + 1):

    # Generate seed for the episode
    episode_seed = np.random.randint(0, 10000)

    observations, _ = env.reset(seed = episode_seed)

    # Restart environment
    observations, _ = env.reset()

    # Initialise dictionaries to store rewards, observations, and actions
    all_observations = {agent_name: [observations[agent_name]] for agent_name in regular_agents.keys()}
    all_actions = {agent_name: [] for agent_name in regular_agents.keys()}
    episode_rewards = {agent_name: [] for agent_name in regular_agents.keys()}

    # Collect data for n_steps (episode length)
    for timestep in range(1, n_steps + 1): # one episode

        actions = {}  # Dictionary to store the current timestep's actions
        
        for agent_name, agent in regular_agents.items():
            state = observations[agent_name]
            # Sample action and opinion
            action, opinion = agent.sample_actions(state)
            actions[agent_name] = (action, opinion) # save action and opinion
        
        # Service providers service availability
        for agent_name, (action, _) in actions.items():            
            # Check if the endpoint for the requested provider is available
            if env.endpoint[agent_name][action] == 1:
                sum_sp_availability[action] += 1

        # Perform actions, determine next state, reward, and termination
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store the rewards, observations, and actions for each agent for the current timestep
        for agent_name in episode_rewards.keys():
            episode_rewards[agent_name].append(rewards[agent_name])
            all_observations[agent_name].append(observations[agent_name])
            all_actions[agent_name].append(actions[agent_name])

    # Store regular agents information, including states, actions, and rewards
    # Process regagent states
    attack_target = 1
    episode_states_pd = process_regagent_states_groups(all_observations, all_actions, providers, malagents, neighbours,
                                   attack_target, episode, n_steps)
    regagent_states_pd = pd.concat([regagent_states_pd, pd.DataFrame(episode_states_pd)], ignore_index=True) # concatenate new data

    # Processing regagent actions and opinions
    mean_selection_rate, mean_expression_rate = process_regagent_actions(all_actions, providers, n_steps)
    actions_history[episode] = mean_selection_rate # add occurrences of each action (as %)
    opinions_history[episode] = mean_expression_rate # add occurrences of each opinion (as %)
    # Process regagent rewards
    sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
    total_rewards[episode] = sum_regagent_rewards
    action_rewards[episode] = service
    opinion_rewards[episode] = feedback
    # Process service provider availability
    sp_availability_episode = process_service_provider_availability(all_actions, providers, sum_sp_availability)
    sp_availability_history[episode] = sp_availability_episode
    sum_sp_availability = np.zeros(len(providers), dtype=int) # reset count to zero

    # Save agents' trust values and actions to csv
    regagent_states_pd.to_csv(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-{episode}-regagents-trust-data.csv'), index = False)
    regagent_states_pd = pd.DataFrame() # Clear the DataFrames for the next set of episodes   

    # Visualise data
    if episode != 0 and episode % vis_freq == 0:
        clear_output(True)
        print("mean reward: %.3f" % (np.mean(total_rewards[-100:-1])))
        
        # Plot 1: Visualise cumulative reward
        plt.figure()
        plt.plot(total_rewards[1:episode], linewidth=0.9, color = 'mediumvioletred', label = "Cumulative reward")
        plt.plot(action_rewards[1:episode], linewidth=0.9, color = 'red', label = "Service reward")
        plt.plot(opinion_rewards[1:episode], linewidth=0.9, color = 'orange', label = "Feedback reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward\nfor regular agents per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-score.png'))
            plt.clf()

        # Plot 2: Visualise social trust in service providers
        colors = cm.tab10(np.arange(nProviders))
        plt.figure()
        for sp in range(nProviders):
            plt.plot(social_trust_history[1:episode, sp], color=colors[sp], linewidth=0.9, label = "Provider  {}".format(sp+1))
        plt.xlabel("Episode")
        plt.ylabel("Average social trust\nin service providers per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-trust.png'))
            plt.clf()

        # Plot 3: Visualise service provider availability
        plt.figure()
        for sp in range(nProviders):
            plt.plot(sp_availability_history[1:episode, sp], color=colors[sp], linewidth=0.9, label = "Provider  {}".format(sp+1))
        plt.xlabel("Episode")
        plt.ylabel("Average service provider availability\nper episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-availability.png'))
            plt.clf()

        # Plot 4.1: Visualise service request rate (%) per episode
        fig, axe = plt.subplots(nrows = 1, ncols = 2, figsize = (15,4))
        for sp in range(nProviders):
            axe[0].plot(actions_history[1:episode, sp], color=colors[sp], linewidth = 0.9,
                label = "Providers {}".format(sp+1))
        axe[0].set_ylim(0,100)
        axe[0].set_xlabel("Episode")
        axe[0].set_ylabel("Average service request rate\nper episode")
        axe[0].grid()
        axe[0].legend(loc=(0.01, 0.50), fontsize='x-small')
        # Plot 4.2: Visualise opinion expression rate (%) per episode
        for o in range(nProviders*2):
            provider_index = o // 2  # Determine which provider this opinion corresponds to
            opinion_type = "Negative" if o % 2 == 0 else "Positive"  # Alternate between negative and positive
            linestyle = "--" if o % 2 == 0 else "-"  # Negative: dashed, Positive: solid
            axe[1].plot(opinions_history[1:episode, o], linestyle, color=colors[provider_index], linewidth=0.9,
                label=f"{opinion_type} opinion on provider {provider_index + 1}")
        axe[1].set_ylim(0,100)
        axe[1].set_xlabel("Episode")
        axe[1].set_ylabel("Average opinion expression rate\nper episode")
        axe[1].grid()
        axe[1].legend(loc=(0.01, 0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-actions-opinions.png'))
            plt.clf()

        plt.show()

# %% [markdown]
# Prepare data dictionary
data_dict = {
    'total_rewards': total_rewards, 'action_rewards': action_rewards, 'opinion_rewards': opinion_rewards,
    'sp0_social_trust': social_trust_history[:, 0], 'sp1_social_trust': social_trust_history[:, 1], 'sp2_social_trust': social_trust_history[:, 2],
    'sp0_request_rate': actions_history[:, 0], 'sp1_request_rate': actions_history[:, 1], 'sp2_request_rate': actions_history[:, 2],
    'op0_expression_rate': opinions_history[:, 0], 'op1_expression_rate': opinions_history[:, 1], 'op2_expression_rate': opinions_history[:, 2],
    'op3_expression_rate': opinions_history[:, 3], 'op4_expression_rate': opinions_history[:, 4], 'op5_expression_rate': opinions_history[:, 5],
    'sp0_availability': sp_availability_history[:, 0], 'sp1_availability': sp_availability_history[:, 1], 'sp2_availability': sp_availability_history[:, 2]
}

# Save to CSV
file_name = f'ch{chapter}-exp{experiment}-{date}-{seed}-regagents-data.csv'
file_path = os.path.join(save_path, file_name)
save_data_to_csv(file_path, data_dict)
