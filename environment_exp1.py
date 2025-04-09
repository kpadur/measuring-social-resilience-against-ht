from gymnasium.spaces import Discrete, Tuple, Box
import numpy as np
import networkx as nx
import torch
import random
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colormaps

class Environment(AECEnv):
    """
    This environment represents a cyber-physical-social system (CPSS).
    """
    
    def __init__(self, nRegAgents, nProviders,
                kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                direct_exp_weight, feedback_adj_term, forgetting_factor):
        """
        The init method takes in environment arguments and creates the environment.

        Attributes are not changed after initialisation.
        """
        self.nRegAgents = nRegAgents
        self.nAgents = nRegAgents
        self.nProviders = nProviders
        self.kappa = kappa
        self.rho = rho
        self.center_up_to_down = center_up_to_down
        self.center_down_to_up = center_down_to_up
        self.end_up_to_down = end_up_to_down
        self.end_down_to_up = end_down_to_up
        self.cost = cost
        self.direct_exp_weight = direct_exp_weight
        self.feedback_adj_term = feedback_adj_term
        self.forgetting_factor = forgetting_factor

        # Create social network as a graph (G = (V, E)) and neighbours dictionary
        # self.social_network, self.neighbours = nx.Graph(), {}
        self.neighbours = {}
        self._form_social_network()

        # Create lists of agent ids (integers)
        self.sn_agents, self.regagents, self.providers = [], [], []
        self._create_agents()

        # Generate separate lists with agent names
        self.agents = [f"regagent{r}" for r in self.regagents]
        # self.agents = [f"regagent{r}" for r in self.sn_agents] # all act as regular agents
        self.possible_agents = self.agents.copy() # all agents that may appear in the environment

        # Initialise regular agents information
        self.opinions_dict, self.opinion_pairs, self.positive_opinions = {}, {}, []
        self._create_regular_agents_info()

        # Observation and action spaces for regular agents  
        self.observation_spaces = {**{f"regagent{ragent}": Box(low=0.0, high=1.0, shape=(self.nProviders,), dtype=np.float32) for ragent in self.regagents}} # trust values
        self.action_spaces = {**{f"regagent{ragent}": Tuple((Discrete(self.nProviders), Discrete(self.nProviders*2))) for ragent in self.regagents}} # service providers, opinions

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        Returns a dictionary of observations and infos (keyed by the agent name)
        """
        # Seed env
        if seed is not None:
            np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

        # Initialise timestep
        self.timestep = 1

        # Reset agents
        self.agents = self.possible_agents.copy()

        # Reset CPS state
        # NOTE: (Assumption) all providers central and endpoint states are initially available
        self.center = {f"defagent{provider}": 1 for provider in self.providers} 
        self.endpoint = {f"regagent{agent}": [1 for _ in range(self.nProviders)] for agent in self.sn_agents}

        # Reset regular agents' information
        self.action_taken = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.opinion_expressed = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.service_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders)] for ragent in self.regagents}} # no service received yet
        self.feedback_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders*2)] for ragent in self.regagents}} # no feedback values yet

        # Reset agents' observations
        # NOTE: (Assumption) regular agents have no previous information about providers (0.5 trust values)
        self.observations = {**{f"regagent{ragent}": [0.5] * self.nProviders for ragent in self.regagents}}

        # Get dummy infos. Necessary for proper environment testing
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        return self.observations, self.infos

    def step(self, actions):
        """
        Receives a dictionary of actions keyed by the agent name. 
        Returns observations, rewards, terminations, truncations, 
        and infos dictionaries keyed by the agent name.
        """
        # Create (new) agent pairs for information spreading
        self.pairs = self._create_pairs_from_graph()

        # Calculate average situational trust
        self.average_situational_trust = self._calculate_average_situational_trust()

        # Give rewards to agents
        rewards = {**{f"regagent{agent}": (self._calculate_service_reward(f"regagent{agent}", actions), 
                                           self._calculate_feedback_reward(f"regagent{agent}", actions)) 
                for agent in self.regagents}}

        # Update observations for agents
        self.observations = {**{f"regagent{agent}": self._calculate_situational_trust(self.service_received[f"regagent{agent}"], 
                                                                                      self.feedback_received[f"regagent{agent}"]) 
               for agent in self.regagents}}
        
        # Update cps state (center, endpoint)
        self._update_cps_state()
        
        # Update timestep
        self.timestep += 1
        
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        return self.observations, rewards, self.terminations, self.truncations, self.infos

    # def render(self, graph_type='both'):
    #     """
    #     Displays rendered frames from the environment for actions and/or opinions.
    #     Parameters:
    #     - graph_type: 'both', 'actions', or 'opinions' to control which graph(s) to display.
    #     """
    #     if graph_type not in ['both', 'actions', 'opinions']:
    #         raise ValueError("Invalid graph_type. Choose 'both', 'actions', or 'opinions'.")

    #     # Create subplots conditionally
    #     if graph_type == 'both':
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    #     else:
    #         fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    #     # Create service provider (action) colour mapping
    #     action_colourmap = plt.cm.get_cmap(colormaps['tab10'], len(self.providers)) 
    #     # Create a new graph for visualisation purposes
    #     combined_graph = nx.Graph(self.social_network)  # Copy the original social network

    #     # Define colours for the original nodes
    #     node_colours_actions = {node: 'gold' if node in self.regagents else 'red' for node in self.social_network.nodes}
    #     node_colours_opinions = node_colours_actions.copy()
    #     # Adding n new separate nodes (representing service providers) with no connections
    #     extra_nodes = [f'd_{provider}' for provider in self.providers]

    #     # Manually position the extra nodes using a circular layout
    #     pos = nx.spring_layout(self.social_network, k=0.5, seed=42)
    #     angle_increment = 1.75 * np.pi / len(extra_nodes)
    #     radius = 1.25  # Distance from the center of the plot

    #     for i, node in enumerate(extra_nodes):
    #         angle = i * angle_increment
    #         pos[node] = [radius * np.cos(angle), radius * np.sin(angle)]

    #     # Add extra nodes to the graph and assign colours
    #     for i, node in enumerate(extra_nodes):
    #         combined_graph.add_node(node)
    #         colour = action_colourmap(i)  # Use 'tab10' colour directly
    #         node_colours_actions[node] = colour
    #         node_colours_opinions[node] = colour

    #     action_opinion_colours = {}
    #     # Assign colours for actions and lighten for negative opinions
    #     for action in range(self.nProviders):
    #         action_colour = to_rgba(action_colourmap(action))
    #         # Lighten the action colour for negative opinion
    #         lightened_colour = tuple((1 - 0.5) * c + 0.5 for c in action_colour[:3]) + (action_colour[3],)
    #         action_opinion_colours[action] = {
    #             'positive': action_colour,
    #             'negative': lightened_colour
    #         }
        
    #     # Render during simulation
    #     if self.timestep > 1:
    #         for agent in self.regagents:
    #             agent_name = f"regagent{agent}"
    #             action = self.action_taken.get(agent_name)
    #             opinion = self.opinion_expressed.get(agent_name)

    #             if action is not None and opinion is not None:
    #                 node_colours_actions[agent] = action_opinion_colours[action]['positive']

    #                 if opinion in self.positive_opinions:
    #                     node_colours_opinions[agent] = action_opinion_colours[action]['positive']
    #                 else:
    #                     node_colours_opinions[agent] = action_opinion_colours[action]['negative']

    #     # Plot for actions if requested
    #     if graph_type in ['both', 'actions']:
    #         ax = ax1 if graph_type == 'both' else ax  # Choose correct axis if 'both' or 'actions'
    #         edge_colors = [combined_graph[u][v].get('color', 'gray') for u, v in combined_graph.edges]
    #         edge_styles = [combined_graph[u][v].get('style', 'solid') for u, v in combined_graph.edges]

    #         nx.draw_networkx_edges(combined_graph, pos=pos, edgelist=combined_graph.edges, edge_color=edge_colors, style=edge_styles, alpha=0.6, ax=ax)
    #         nx.draw_networkx_nodes(combined_graph, pos=pos, node_color=[node_colours_actions.get(node, 'mediumvioletred') for node in combined_graph.nodes], node_size=[300 if node in self.social_network.nodes else 2000 for node in combined_graph.nodes], ax=ax)
    #         ax.text(0, 1, "A", transform=ax.transAxes, fontsize=20, verticalalignment='top')
    #         ax.axis('off')

    #     # Plot for opinions if requested
    #     if graph_type in ['both', 'opinions']:
    #         ax = ax2 if graph_type == 'both' else ax  # Choose correct axis if 'both' or 'opinions'
    #         edge_colors = [combined_graph[u][v].get('color', 'gray') for u, v in combined_graph.edges]
    #         edge_styles = [combined_graph[u][v].get('style', 'solid') for u, v in combined_graph.edges]

    #         nx.draw_networkx_edges(combined_graph, pos=pos, edgelist=combined_graph.edges, edge_color=edge_colors, style=edge_styles, alpha=0.6, ax=ax)
    #         nx.draw_networkx_nodes(combined_graph, pos=pos, node_color=[node_colours_opinions.get(node, 'mediumvioletred') for node in combined_graph.nodes], node_size=[300 if node in self.social_network.nodes else 2000 for node in combined_graph.nodes], ax=ax)
    #         ax.text(0, 1, "B", transform=ax.transAxes, fontsize=20, verticalalignment='top')
    #         ax.axis('off')

    #     plt.tight_layout()
    #     return fig  # Return the figure without showing it

    @property
    def num_agents(self) -> int: # length of the agent list
        return len(self.agents)

    @property
    def max_num_agents(self) -> int: # length of possible agents list
        return len(self.possible_agents)

    def _form_social_network(self):
        """
        Creates social network graph as a small-world network with high clustering and low average path length,
        here, Watts-Strogatz graph.
        """
        # self.graph = nx.watts_strogatz_graph(self.nAgents, self.kappa, self.rho)
        # self.social_network.add_nodes_from(self.graph)
        # self.social_network.add_edges_from(self.graph.edges)
        # self.neighbours = nx.to_dict_of_lists(self.social_network)
        # neighbours generated with seed 743 (preset)
        self.neighbours = {0: [1, 109, 2, 108, 3, 107], 1: [0, 2, 3, 109, 4, 108], 2: [0, 1, 3, 4, 5], 3: [0, 1, 2, 5, 6, 54], 4: [1, 2, 5, 6, 7], 5: [2, 3, 4, 6, 7, 8], 6: [3, 4, 5, 7, 8, 9], 7: [4, 5, 6, 8, 9, 10], 8: [5, 6, 7, 9, 10, 11], 9: [6, 7, 8, 10, 11, 12, 36], 10: [7, 8, 9, 11, 12, 13, 109],\
                        11: [8, 9, 10, 12, 13, 14], 12: [9, 10, 11, 13, 14, 15], 13: [10, 11, 12, 14, 15, 16], 14: [11, 12, 13, 15, 16, 17], 15: [12, 13, 14, 16, 17, 18], 16: [13, 14, 15, 17, 18, 19], 17: [14, 15, 16, 18, 19, 20], 18: [15, 16, 17, 19, 20, 21], 19: [16, 17, 18, 20, 21, 22, 103], 20: [17, 18, 19, 21, 22, 23], \
                        21: [18, 19, 20, 22, 23, 24], 22: [19, 20, 21, 23, 24, 25], 23: [20, 21, 22, 24, 25, 26], 24: [21, 22, 23, 25, 26, 27], 25: [22, 23, 24, 26, 27, 28], 26: [23, 24, 25, 27, 28, 29], 27: [24, 25, 26, 28, 29, 30], 28: [25, 26, 27, 29, 30, 31], 29: [26, 27, 28, 30, 31, 32], 30: [27, 28, 29, 31, 32, 33, 67], \
                            31: [28, 29, 30, 32, 33, 34], 32: [29, 30, 31, 33, 34, 35], 33: [30, 31, 32, 34, 35, 47], 34: [31, 32, 33, 35, 36, 37], 35: [32, 33, 34, 36, 37, 38], 36: [9, 34, 35, 37, 39], 37: [34, 35, 36, 38, 39, 40], 38: [35, 37, 39, 40, 41], 39: [36, 37, 38, 41, 42, 109], 40: [37, 38, 41, 42, 43, 88], \
                            41: [38, 39, 40, 42, 43, 44], 42: [39, 40, 41, 43, 44, 45], 43: [40, 41, 42, 44, 45, 46], 44: [41, 42, 43, 45, 46, 47], 45: [42, 43, 44, 46, 47, 48], 46: [43, 44, 45, 47, 48, 49], 47: [33, 44, 45, 46, 48, 49, 50], 48: [45, 46, 47, 49, 50, 89], 49: [46, 47, 48, 50, 51, 52, 99], 50: [47, 48, 49, 51, 52, 53], \
                            51: [49, 50, 52, 53, 54], 52: [49, 50, 51, 53, 54, 55], 53: [50, 51, 52, 54, 55, 56], 54: [3, 51, 52, 53, 55, 56, 57, 90], 55: [52, 53, 54, 56, 57, 58], 56: [53, 54, 55, 57, 58, 59], 57: [54, 55, 56, 58, 59, 60], 58: [55, 56, 57, 59, 60, 61], 59: [56, 57, 58, 60, 61, 62], 60: [57, 58, 59, 62, 63, 69], \
                            61: [58, 59, 62, 63, 64], 62: [59, 60, 61, 63, 64, 65], 63: [60, 61, 62, 64, 65, 66], 64: [61, 62, 63, 65, 66, 67], 65: [62, 63, 64, 66, 67, 68], 66: [63, 64, 65, 67, 68, 69], 67: [30, 64, 65, 66, 69, 70], 68: [65, 66, 69, 70, 71], 69: [60, 66, 67, 68, 70, 71, 72], 70: [67, 68, 69, 71, 72, 73], \
                            71: [68, 69, 70, 72, 73, 74], 72: [69, 70, 71, 73, 74, 75], 73: [70, 71, 72, 74, 75, 76], 74: [71, 72, 73, 75, 76, 77], 75: [72, 73, 74, 76, 77, 78], 76: [73, 74, 75, 78, 79, 108], 77: [74, 75, 78, 80, 102], 78: [75, 76, 77, 79, 80, 81], 79: [76, 78, 80, 81, 82], 80: [77, 78, 79, 81, 82, 83], \
                            81: [78, 79, 80, 82, 83, 84], 82: [79, 80, 81, 83, 84, 85], 83: [80, 81, 82, 84, 85, 86], 84: [81, 82, 83, 85, 86, 87], 85: [82, 83, 84, 86, 87, 88], 86: [83, 84, 85, 87, 88, 89], 87: [84, 85, 86, 88, 89, 90], 88: [40, 85, 86, 87, 89, 90], 89: [48, 86, 87, 88, 90, 91, 92], 90: [54, 87, 88, 89, 92, 93], \
                            91: [89, 92, 93, 94], 92: [89, 90, 91, 93, 94, 95], 93: [90, 91, 92, 94, 95, 96], 94: [91, 92, 93, 95, 96, 97], 95: [92, 93, 94, 96, 97, 98], 96: [93, 94, 95, 97, 98, 99], 97: [94, 95, 96, 98, 99, 100], 98: [95, 96, 97, 99, 100, 101], 99: [49, 96, 97, 98, 100, 101], 100: [97, 98, 99, 101, 102, 103], \
                            101: [98, 99, 100, 102, 103, 104], 102: [77, 100, 101, 103, 104, 105], 103: [19, 100, 101, 102, 105, 106], 104: [101, 102, 105, 106, 107], 105: [102, 103, 104, 106, 107, 108], 106: [103, 104, 105, 107, 108, 109], 107: [0, 104, 105, 106, 108, 109], 108: [0, 1, 76, 105, 106, 107, 109], 109: [0, 1, 10, 39, 106, 107, 108]}


    def _create_agents(self):
        """
        Creates lists of regular agents, malicious agents (attackers), and service providers (defenders)
        """
        # Create list of all agents
        self.sn_agents = [agent for agent in range(self.nAgents)]

        # # Specify list of malicious agents from all agents
        # # NOTE: (Assumption) malicious agents location in the social network is selected based on their centrality 
        # # and distance to other malicious agents (not connected to each other)

        # # Compute the betweenness centrality of social network
        # centrality = nx.betweenness_centrality(self.social_network)
        # # Sort nodes by centrality score
        # sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        # for node, centrality in sorted_nodes:
        #     # Check if this node is a neighbor to any previously selected top nodes
        #     if all(not self.social_network.has_edge(node, top_node) for top_node in self.malagents):
        #         self.malagents.append(node)
        #     if len(self.malagents) == self.nMalAgents:
        #         break
        
        # Specify list of regular agents from all remaining agents
        self.regagents = [regagent for regagent in self.sn_agents]
        # self.regagents = [regagent for regagent in self.sn_agents if regagent not in self.malagents]

        # Create list of service providers
        self.providers = [provider for provider in range(self.nProviders)]

    def _create_regular_agents_info(self):
        """
        Create opinions dictionery, e.g., {0:[-1,0], 1:[1,0], 2:[-1,1], 3:[1,1], 4:[-1,2], 5:[1,2]},
        opinion pairs, e.g., {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}, and positiove opinions, e.g., [1,3,5]
        """
        # Opinions dictionary
        for i in range(self.nProviders*2):
            if i%2==0: self.opinions_dict[i] = [-1, i//2]
            else: self.opinions_dict[i] = [1, i//2]
        # Opinion pairs
        for j in range(0, self.nProviders*2, 2):
            self.opinion_pairs[j] = j+1
            self.opinion_pairs[j+1] = j
        # Generate list of positive opinions
        for k, v in self.opinions_dict.items():
            if v[0] == 1: self.positive_opinions.append(k)

    def _create_pairs_from_graph(self):
        """
        Determine agent pairs from social network graph that can interact with each other
        """
        nodes = self.sn_agents
        pairs = []
        # Copy the list of all nodes to remaining nodes to be able to modify it
        remaining_nodes = nodes.copy()

        # Randomise remaining_nodes (instead of the original nodes list)
        random.shuffle(remaining_nodes)
        # While there are nodes left in the remaining_nodes list
        while remaining_nodes:
            node = remaining_nodes.pop()
            # neighbors = list(self.social_network.neighbors(node))
            neighbors = self.neighbours[node]
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor in remaining_nodes:
                    pairs.append((node, neighbor))
                    remaining_nodes.remove(neighbor)
                    break
        return pairs

    def _get_interaction(self, agent):
        """
        Determine the agent's interaction partner
        return:     int, agent's interaction partner
        """
        interaction_dict = dict(self.pairs)
        # Check if the agent is a key in the dictionary
        if agent in interaction_dict:
            return interaction_dict[agent]
        # Check if the agent is a value in the dictionary
        for key, value in interaction_dict.items():
            if value == agent:
                return key
        # If not found, return -1
        return -1

    def _update_cps_state(self):
        """
        Updates service providers' behaviour based on Markov model for the next time step;
        center and endpoint as dictionaries 
        """
        for provider in self.providers:
            provider_name = f"defagent{provider}"
            # Determine provider's central transition matrix (from 0 to 0, from 0 to 1, from 1 to 0, from 1 to 1)
            center_matrix = [[1 - self.center_down_to_up[provider], self.center_down_to_up[provider]],\
                             [self.center_up_to_down[provider], 1 - self.center_up_to_down[provider]]]

            # Determine the next state based on the transition probabilities
            center_state = self.center[provider_name]
            next_center = random.choices([0, 1], weights=center_matrix[center_state])[0]
            self.center[provider_name] = next_center

        for provider in self.providers:
            provider_name = f"defagent{provider}"
            # Determine provider's endpoint transition matrix (from 0 to 0, from 0 to 1, from 1 to 0, from 1 to 1)
            end_matrix = [[1 - self.end_down_to_up[provider], self.end_down_to_up[provider]],\
                          [self.end_up_to_down[provider], 1 - self.end_up_to_down[provider]]]

            for agent in self.sn_agents: # Generate service for each agent
                agent_name = f"regagent{agent}"
                # Generate service for each agent
                if self.center[provider_name] == 1: 
                    # Generate endpoint only when center is available
                    endpoint_state = self.endpoint[agent_name][provider]
                    next_endpoint = random.choices([0, 1], weights=end_matrix[endpoint_state])[0]
                    self.endpoint[agent_name][provider] = next_endpoint
                else:
                    self.endpoint[agent_name][provider] = 0

    def _calculate_service_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on service and cost
        return:     float
        """
        provider = actions[agent_name][0] # agent's action (sp id) for this time step

        # Determine provider's endpoint state for this specific agent
        endpoint_state = self.endpoint[agent_name][provider] # 1: available, 0: unavailable

        # Check if service request was unavailable
        if endpoint_state == 1:
            service_reward = 1
        else:
            service_reward = -1

        # Determine cost associated with the action
        if self.action_taken[agent_name] is None: # agent has not taken any actions yet
            reward = service_reward
        elif self.action_taken[agent_name] == provider: # agent stayed with the same provider
            reward = service_reward + self.cost[provider]
        else: # agent changed provider
            reward = service_reward - self.cost[provider]

        # Change action to the most recent action
        self.action_taken[agent_name] = provider

        # Save requested service state and time step for this agent
        self.service_received[agent_name][provider].append((self.timestep, service_reward))
        return float(reward)

    def _calculate_feedback_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on feedback from neighbour
        return:     float
        """
        # Determine interacting agents
        agent = int(agent_name.replace("regagent", ""))  # agent ID
        neighbour = self._get_interaction(agent)  # neighbour ID
        opinion = actions[agent_name][1]  # Agent's (expressed) opinion
        self.opinion_expressed[agent_name] = opinion  # Save opinion for tracking

        # Identify the service provider associated with the expressed opinion
        agent_opinion_value = self.opinions_dict[opinion][0]  # Opinion value (-1 or 1)
        agent_opinion_sp = self.opinions_dict[opinion][1]  # Service provider ID
        

        # Determine feedback from the neighbour
        if neighbour == -1:  # Agent has no contact this timestep
            feedback = 0
        else:
            neighbour_name = f"regagent{neighbour}"
            neighbour_trust = self.observations[neighbour_name][agent_opinion_sp]  # neighbour's trust for this provider
            # Determine feedback based on dynamic trust threshold
            if neighbour_trust >= self.average_situational_trust[agent_opinion_sp]:
                feedback = 1 * agent_opinion_value # neighbour has pos opinion
            else:
                feedback = -1 * agent_opinion_value # neighbour has neg opinion
        
        # Give additional support for opinion that agent itself has strong direct experience with
        if actions[agent_name][0] == agent_opinion_sp:
            support = feedback + self.feedback_adj_term
        else:
            support = feedback - self.feedback_adj_term

        # Save feedback received
        if feedback != 0:
            self.feedback_received[agent_name][opinion].append((self.timestep, feedback))
        return float(support)
    
    def _calculate_situational_trust(self, service_states, feedback_values):
        """
        Calculates situational trust for regular agent based on service states.
        The objective of a trust assessment mechanism is to allow the best agent selection
        in the light of the uncertainties linked to dynamic agent behaviour.
        return:     float
        """
        current_time = self.timestep

        # Analyse direct experiences
        pos_exps = np.zeros(self.nProviders, dtype = float)
        neg_exps = np.zeros(self.nProviders, dtype = float)

        for provider in range(self.nProviders):
            service_states[provider].sort(key=lambda x: x[0]) # should be sorted already

            # Separate positive experiences from negative experiences
            for i, reward in service_states[provider]:
                if reward > 0:
                    pos_exps[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    neg_exps[provider] += 1 * self.forgetting_factor**(current_time-i)
        # Calculate direct experience score for all service providers
        direct_exp_score = (pos_exps + 1)/(pos_exps + neg_exps + 2)

        # Analyse witness information
        pos_wit = np.zeros(self.nProviders, dtype = float)
        neg_wit = np.zeros(self.nProviders, dtype = float)

        for opinion in self.positive_opinions:
            alt_opinion = self.opinion_pairs[opinion]
            provider = self.opinions_dict[opinion][1]

            # Separate positive feedback from negative feedback
            for i, reward in feedback_values[opinion]:
                if reward > 0:
                    pos_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    neg_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
            # Deal with alternative opinion
            for i, reward in feedback_values[alt_opinion]:
                if reward > 0:
                    neg_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    pos_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
        # Calculate feedback score for all service providers
        feedback_score = (pos_wit + 1)/(pos_wit + neg_wit + 2)

        # Calculate situational trust
        situational_trust = self.direct_exp_weight * direct_exp_score + (1 - self.direct_exp_weight) * feedback_score
        return situational_trust
    
    def _calculate_average_situational_trust(self):
        """
        Calculate average situational trust and dynamic thresholds for each service provider.
        Returns:
            mean_trusts (np.ndarray): Mean trust values for each provider.
            dynamic_thresholds (np.ndarray): Dynamic thresholds for each provider.
        """
        # Initialise arrays to collect trust values for each provider
        situational_trust_values = [[] for _ in range(self.nProviders)]

        # Collect situational trust values for each provider
        for agent_trust in self.observations.values():
            for provider in range(self.nProviders):
                situational_trust_values[provider].append(agent_trust[provider])

        # Compute mean and standard deviation for each provider
        mean_trusts = np.zeros(self.nProviders, dtype=float)
        std_trusts = np.zeros(self.nProviders, dtype=float)
        for provider in range(self.nProviders):
            trust_values = situational_trust_values[provider]
            mean_trusts[provider] = np.mean(trust_values)
            std_trusts[provider] = np.std(trust_values)

        # Compute dynamic thresholds
        dynamic_thresholds = mean_trusts - std_trusts

        return dynamic_thresholds

