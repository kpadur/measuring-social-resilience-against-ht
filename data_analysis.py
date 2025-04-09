import pandas as pd
import numpy as np
import random

def moving_average(data): #
    window_size = 200
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def populate_attack_strategy(strategy_df, attack_strategy, warmup_time): #
    stage_actions = {
        'recon': 0,
        'cyber': (1, 4),
        'disinfo': (2, 5),
        'comb': (3, 6),
        'cyber2': (1, 4),
        'disinfo2': (2, 5),
        'terminate': 7
    }
    # Loop through each strategy
    for idx, row in strategy_df.iterrows():
        current_time = warmup_time  # Initial 100 timesteps are idle
        
        # Initial warmup time there are not actions
        attack_strategy[idx, :warmup_time+1] = 0
        
        # Reconnaissance stage
        attack_strategy[idx, warmup_time+1] = stage_actions['recon']  # Recon stage starts at timestep 201
        current_time += 1

        # Populate each stage based on the order and durations
        stages = ['cyber', 'disinfo', 'comb', 'cyber2', 'disinfo2']
        for stage in stages:
            duration = row[stage]
            if duration > 0:
                first_action, subsequent_action = stage_actions[stage]
                # First action of the stage
                attack_strategy[idx, current_time] = first_action
                current_time += 1
                # Subsequent actions of the stage
                attack_strategy[idx, current_time:current_time + duration - 1] = subsequent_action
                current_time += duration - 1

        # Terminate action after all stages
        attack_strategy[idx, current_time:] = stage_actions['terminate']

    return attack_strategy

def process_regagent_states_groups(all_observations, all_actions, providers, malagents, neighbours,
                                   attack_target, step, n_steps): #
    # Only process regagent states
    regagent_observations = {agent: values for agent, values in all_observations.items() if agent.startswith('regagent')}
    # Determine only regagent actions
    regagent_actions = {agent: values for agent, values in all_actions.items() if agent.startswith('regagent')}
    # Initialise empty groups
    groups = {"group0": [[] for _ in range(n_steps)],
            "group1": [[] for _ in range(n_steps)],
            "group2": [[] for _ in range(n_steps)],
            "group3": [[] for _ in range(n_steps)]}
    
    # Allocate agents into groups
    for t in range(n_steps):
        for regagent_name in regagent_actions:
            regagent_id = int(regagent_name.replace("regagent", ""))

            if regagent_actions[regagent_name][t][0] == attack_target and\
                np.any(np.isin(neighbours[regagent_id], malagents)):
                groups["group3"][t].append(regagent_name)
            elif regagent_actions[regagent_name][t][0] == attack_target:
                groups["group1"][t].append(regagent_name)
            elif np.any(np.isin(neighbours[regagent_id], malagents)):
                groups["group2"][t].append(regagent_name)
            else:
                groups["group0"][t].append(regagent_name)

    # Initialise empty matrices for trust information
    trust_sum = np.zeros((n_steps, len(groups), len(providers)))
    avg_trust = np.zeros((n_steps, len(groups), len(providers)))

    # Calculate average trust values about each service provider for each group at each timestep
    for group_id, values in enumerate(groups.values()):
        for t in range(n_steps):
            for agent_name in values[t]:
                trust_sum[t][group_id] += regagent_observations[agent_name][t]
            if len(values[t]) != 0:
                avg_trust[t][group_id] = trust_sum[t][group_id] / len(values[t])

    # Get shape of avg_trust
    n_steps, n_groups, n_providers = avg_trust.shape
    # Generate column names
    column_names = [f"group{group}_trust_in_sp{sp}" for group in range(n_groups) for sp in range(n_providers)]
    # Reshape avg_trust to 2D array
    reshaped_avg_trust = avg_trust.reshape(n_steps, -1)
    # Create DataFrame
    regagent_states_df = pd.DataFrame(data=reshaped_avg_trust, columns=column_names)
    # Add 'episode' and 'timestep' columns
    regagent_states_df.insert(0, 'episode', step)
    regagent_states_df.insert(1, 'timestep', range(n_steps))

    return regagent_states_df

def process_regagent_actions(all_actions, providers, n_steps): #
    """
    Calculate the average service request rate (per episode) for each service provider and
    the average opinion experssing rate (per episode) for each service provider.
    """
    # Analyse action and opinion counts
    all_regagent_actions = {
        agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('regagent')
        }
    regagent_action_counts = [0] * len(providers)
    regagent_opinion_counts = [0] * len(providers)*2

    for actions_list in all_regagent_actions.values():
        for action, opinion in actions_list:
            regagent_action_counts[action] += 1
            regagent_opinion_counts[opinion] += 1

    mean_selection_rate = np.array(regagent_action_counts) * 100/ (len(all_regagent_actions)*n_steps)
    mean_expression_rate = np.array(regagent_opinion_counts) * 100/ (len(all_regagent_actions)*n_steps)

    return mean_selection_rate, mean_expression_rate

def process_regagent_rewards(episode_rewards): #
    # Determine total rewards
    regagent_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('regagent')}
    sum_regagent_rewards = sum([sum(r) for rewards in regagent_rewards.values() for r in rewards])
    # Determine service rewards
    service = sum([r[0] for rewards in regagent_rewards.values() for r in rewards])
    # Determine feedback rewards
    feedback = sum([r[1] for rewards in regagent_rewards.values() for r in rewards])
    return sum_regagent_rewards, service, feedback

def process_service_provider_availability(all_actions, providers, sum_sp_availability): #
    """
    Calculate average service availability rate per episode for each service provider (when requested).
    """
    # Analyse action and opinion counts
    all_regagent_actions = {
        agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('regagent')
        }
    regagent_action_counts = np.zeros(len(providers), dtype=int)

    for actions_list in all_regagent_actions.values():
        for action, _ in actions_list:
            regagent_action_counts[action] += 1
    avg_service_availability = sum_sp_availability / regagent_action_counts

    return avg_service_availability

def process_attacker_rewards(all_actions, episode_rewards): #
    # Determine attacker rewards
    attack_actions = {agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('malagent')}
    attacker_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('malagent')}
    attack_stage_actions = np.array([action[0] for action in attack_actions.values() for action in action])
    attack_stage_rewards = np.array([reward[0] for reward in attacker_rewards.values() for reward in reward])
    # Calculate recon rewards
    recon_rewards = attack_stage_rewards[np.where(attack_stage_actions == 0)]
    # Determine term rewards
    term_rewards = attack_stage_rewards[np.where(attack_stage_actions == 7)]
    # Bot (cyberattack) rewards
    bot_rewards = np.array([reward[1] for reward in attacker_rewards.values() for reward in reward])
    # Contact rewards
    contact_rewards = 0
    for _, (attack_stage_reward, bot_reward, misinfo_reward) in enumerate(attacker_rewards["malagent"]):
        contact_rewards += sum(misinfo_reward.values())
    return np.sum(attack_stage_rewards), np.sum(recon_rewards), np.sum(term_rewards), np.sum(bot_rewards), contact_rewards

def process_attacker_actions(all_actions, n_cyber_actions, n_misinfo_actions): #
    # Collect attacker's data at every step for visualisation
    all_attack_actions = {agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('malagent')}
    # Cyberattack actions
    attack_actions = np.array([action[1] for actions in all_attack_actions.values() for action in actions]) + 1 # so that -1 would account to 0
    # Misinformation actions
    misinfo_actions = np.array([action[2] for actions in all_attack_actions.values() for action in actions])
    contact_actions = np.array([val for d in misinfo_actions for val in d.values()])

    # Cyberattack
    bot_counts = np.zeros(n_cyber_actions+1, dtype=np.int64)
    if np.any(attack_actions != 0):
        index = np.where(attack_actions != 0)
        attack_actions = attack_actions[index]
        unique_actions1, action_counts1 = np.unique(attack_actions, return_counts=True)
        bot_counts[unique_actions1] = action_counts1
    # Misinfo
    misinfo_counts = np.zeros(n_misinfo_actions+1, dtype = np.int64)
    if np.any(contact_actions != -1):
        index = np.where(contact_actions != -1)
        contact_actions = contact_actions[index]
        unique_actions2, action_counts2 = np.unique(contact_actions, return_counts=True)
        misinfo_counts[unique_actions2] = action_counts2

    stage_actions = np.array([action[0] for actions in all_attack_actions.values() for action in actions])
    episode_attack_order, episode_count_actions_per_stage = determine_attack_order_and_length(stage_actions)
    return bot_counts, misinfo_counts, episode_attack_order, episode_count_actions_per_stage

def determine_attack_order_and_length(stage_actions): #
    # Stage actions is an array of actions made by the attacker in each attack stage that determines how attacker 'moves' between stages
    # Actions
    # Attack stage
    values_to_stop = 7 # terminate action
    for index, element in enumerate(stage_actions):
        if element == values_to_stop:
            stage_actions = stage_actions[:index+1]
            break

    episode_attack_order = np.zeros(4+2, dtype = int)
    episode_count_actions_per_stage = np.zeros(4+2, dtype = int)
    for action in stage_actions:
        # Create timeline of actions
        max_value = max(episode_attack_order)
        if action in [0] and episode_attack_order[0] == 0:
            episode_attack_order[0] = max_value + 1
        elif action in [1]:
            # First time cyberattack
            if episode_attack_order[1] == 0: episode_attack_order[1] = max_value + 1
            # Return to cyberattack
            else: episode_attack_order[4] = max_value + 1
        elif action in [2]:
            # First time misinformation
            if episode_attack_order[2] == 0: episode_attack_order[2] = max_value + 1
            # Return to misinformation
            else: episode_attack_order[5] = max_value + 1
        elif action in [3] and episode_attack_order[3] == 0:
            episode_attack_order[3] = max_value + 1
        # Count actions
        index = np.where(episode_attack_order== max(episode_attack_order))[0][0]
        if action not in [7]:
            episode_count_actions_per_stage[index] += 1
    # return stage_actions, attack_order, count_actions_per_stage
    return episode_attack_order, episode_count_actions_per_stage

def process_defender_rewards(episode_rewards, providers): #
    defenders_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('defagent')}
    filter_rewards = np.zeros(len(providers))
    answer_rewards = np.zeros(len(providers))
    for defender in providers:
        filter_sum, answer_sum = 0, 0
        for _, (f, a) in enumerate(defenders_rewards[f"defagent{defender}"]):
            filter_sum += np.sum(f)
            answer_sum += np.sum(a)
        filter_rewards[defender] = filter_sum
        answer_rewards[defender] = answer_sum
    return filter_rewards, answer_rewards

def process_attack_strategy(attack_order, count_timesteps_per_stage):
    # Analyse attack order data
    # Initialise an empty list to collect data
    attack_order_data = []

    # Expand each tuple into a list of (step, attack_order) pairs
    for step, orders in attack_order:
        row = {'step': step}
        for i, order in enumerate(orders):
            row[f'attack_stage_{i}'] = order
        attack_order_data.append(row)

    # Create a DataFrame from the collected data
    attack_order_df = pd.DataFrame(attack_order_data)

    # Reshaping the DataFrame to have one row per step
    attack_order_df = attack_order_df.groupby('step').first().reset_index()

    # Fill NaN values with 0 and cast to integer
    attack_order_df = attack_order_df.fillna(0).astype(int)

    # Analyse attack actions count data
    attack_count_data = []

    for step, counts in count_timesteps_per_stage:
        row = {'step': step}
        for i, count in enumerate(counts):
            row[f'attack_stage_{i}'] = count
        attack_count_data.append(row)
    
    # Create a DataFrame from the collected data
    count_attack_actions_df = pd.DataFrame(attack_count_data)

    # Reshaping the DataFrame to have one row per step
    count_attack_actions_df = count_attack_actions_df.groupby('step').first().reset_index()

    # Fill NaN values with 0 and cast to integer
    count_attack_actions_df = count_attack_actions_df.fillna(0).astype(int)

    return attack_order_df, count_attack_actions_df
