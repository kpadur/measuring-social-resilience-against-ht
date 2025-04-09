from gymnasium.spaces import Discrete, MultiDiscrete, Tuple, MultiBinary, Dict, Box
import numpy as np
import networkx as nx
import torch
import random
from pettingzoo import AECEnv
import re
import matplotlib.pyplot as plt

class Environment(AECEnv):
    """
    This environment represents a cyber-physical-social system (CPSS).
    """
    
    def __init__(self, nRegAgents, nMalAgents, nProviders,
                kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                direct_exp_weight, feedback_adj_term, forgetting_factor):
        """
        The init method takes in environment arguments and creates the environment.

        Attributes are not changed after initialisation.
        """
        self.nRegAgents = nRegAgents
        self.nMalAgents = nMalAgents
        self.nAgents = nRegAgents + nMalAgents
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
        # self.social_network = nx.Graph()
        self.neighbours = {}
        self._form_social_network()

        # Create lists of agent ids (integers)
        self.sn_agents, self.regagents, self.malagents, self.providers = [], [], [], []
        self._create_agents()

        # Generate separate lists with agent names
        regagent_ids = [f"regagent{r}" for r in self.regagents]
        defagent_ids = [f"defagent{d}" for d in self.providers]
        malagent_ids = ["malagent"]

        self.agents = regagent_ids + defagent_ids + malagent_ids # agents active at any given time
        self.possible_agents = self.agents.copy() # all agents that may appear in the environment

        # Identify malicious agents (array of False and True)
        self.malagent_indices = np.array([agent in self.malagents for agent in self.sn_agents])
        # Identify regular agents
        self.regagent_indices = np.array([agent in self.regagents for agent in self.sn_agents])

        # Initialise regular agents information
        self.opinions_dict, self.opinion_pairs, self.positive_opinions, self.negative_opinions = {}, {}, [], []
        self._create_regular_agents_info()
        # Initialise malicious agents information
        self.new_stage_dict, self.mcontacts, self.mcontacts_dict = {}, set(), {}
        self._create_attackers_info()

        # Observation and action spaces for each agent
        self.observation_spaces = {
            **{f"regagent{ragent}": Box(low=0.0, high=1.0, shape=(self.nProviders,), dtype=np.float32) for ragent in self.regagents}, # situational trust in service providers
            **{f"defagent{provider}": Tuple((MultiBinary(len(self.sn_agents)), MultiDiscrete([3] * len(self.sn_agents)))) for provider in self.providers}, # service requests, opinions (0: no opinion, 1: positive, 2: negative)
            **{f"malagent": Tuple([Discrete(2) for _ in range(len(self.sn_agents))] + [Discrete(len(self.new_stage_dict))])}
        }
        # Action spaces for each agent
        self.action_spaces = {
            **{f"regagent{ragent}": Tuple((Discrete(self.nProviders), Discrete(self.nProviders*2))) for ragent in self.regagents}, # service providers, opinion
            **{f"defagent{provider}": Tuple((MultiBinary(len(self.sn_agents)), MultiBinary(len(self.sn_agents)))) for provider in self.providers}, # service requests, opinions in sn
            **{f"malagent": Tuple([Discrete(8), Discrete(len(self.malagents)), Dict({
                f"malagent{malagent}": Discrete(len(self.mcontacts)) for malagent in self.malagents})])} # attack stage action, botnet size, sn actions as dict ({malagent: action, ...})
        }

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
        self.actions_taken = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.opinion_expressed = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.service_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders)] for ragent in self.regagents}} # no service received yet
        self.feedback_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders*2)] for ragent in self.regagents}} # no feedback values yet

        # Reset defenders' information
        # NOTE: (Assumption) defenders have no previous information about agents
        self.defenders_observations = {**{f"defagent{provider}": ([0] * len(self.sn_agents), [0] * len(self.sn_agents)) for provider in self.providers}}
            
        # Reset attackers' information; NOTE: (Assumption) attack target is the first provider
        self.attacked_provider = 1 # not scalable, works for now
        self.misinfo_opinion = [key for key, val in self.opinions_dict.items() if val == [-1, self.attacked_provider]][0]
        self.misinfo_alt_opinion = [key for key, val in self.opinions_dict.items() if val == [1, self.attacked_provider]][0]

        # Reset agents' observations
        self.observations = {
            **{f"regagent{ragent}": [0.5] * self.nProviders for ragent in self.regagents}, # changed scale to be between 0 and 1
            **{f"defagent{provider}": ([0] * len(self.sn_agents), [0] * len(self.sn_agents)) for provider in self.providers}, # zero service requests and opinions
            **{f"malagent": (1,) * len(self.sn_agents) + (0,)} # NOTE: (Assumption) attacker is in the reconnaissance stage
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        return self.observations, self.infos

    def step(self, actions):
        """
        Receives a dictionary of actions keyed by the agent name. 
        Returns the observation dictionary, reward dictionary, terminated dictionary, 
        truncated dictionary and info dictionary, where each dictionary is keyed by the agent.
        """
        # Create (new) agent pairs for information spreading
        # NOTE: (Assumption) pairs are generated based on attackers' actions
        self.pairs = self._create_pairs_from_graph(actions)
        # Create a random list of regular agents (size equals botnet size minus number of blocked attackers) who were affected by a denial of service attack
        self.dos_agents = self._create_dos_affected_agents(actions) # list of agents (integers not agent names)

        # Calculate average situational trust
        self.average_situational_trust = self._calculate_average_situational_trust()

        # Give rewards to agents
        rewards = {
            **{f"regagent{agent}": (
                self._calculate_service_reward(f"regagent{agent}", actions),
                self._calculate_feedback_reward(f"regagent{agent}", actions)
                ) for agent in self.regagents},
            **{f"defagent{agent}": (
                self._calculate_filtering_rewards(f"defagent{agent}", actions), 
                self._calculate_sn_answer_rewards(f"defagent{agent}", actions)
                ) for agent in self.providers},
            **{f"malagent": self._calculate_attacker_rewards(actions)} # Tuple (attack stage action reward, cyberattack reward, misinfo reward)
        }

        # Update cps state (center, endpoint)
        self._update_cps_state()

        # Update observations for agents
        self.observations = {
            **{f"regagent{agent}": self._calculate_situational_trust(self.service_received[f"regagent{agent}"], self.feedback_received[f"regagent{agent}"]) 
            for agent in self.regagents},
            **{f"defagent{agent}": (
                (self.defenders_observations[f"defagent{agent}"][0], self.defenders_observations[f"defagent{agent}"][1]))
                for agent in self.providers},
            **{f"malagent": 
            (self._determine_service_availability(f"defagent1")) + (self._update_attack_stage(actions["malagent"][0]),)
            }
        }
        
        # Update timestep
        self.timestep += 1
        
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        return self.observations, rewards, self.terminations, self.truncations, self.infos

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
        # Preset malagents
        self.malagents = [54, 30, 103, 109, 47, 76, 77, 99, 9, 69]
        # Specify list of regular agents from all remaining agents
        self.regagents = [regagent for regagent in self.sn_agents if regagent not in self.malagents]

        # Create list of service providers
        self.providers = [provider for provider in range(self.nProviders)]

    def _create_regular_agents_info(self):
        """
        Create opinions dictionery, e.g., {0:[-1,0], 1:[1,0], 2:[-1,1], 3:[1,1], 4:[-1,2], 5:[1,2]},
        opinion pairs, e.g., {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}, positiove opinions, e.g., [1,3,5],
        and negative opinions, e.g.,  [0,2,4].
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
        # Generate list of negative opinions
        for k, v in self.opinions_dict.items():
            if v[0] == -1: self.negative_opinions.append(k)

    def _create_attackers_info(self):
        """
        Creates attackers' information
        """
        # Create dictionary of stage actions - which actions can be taken in each attack stage
        self.stage_actions_dict = {0:[0,1,2,3,7], 1:[2,3,4,7], 2:[4,7], 3:[4,7], 4:[4,7],\
                                            5:[1,3,5,7], 6:[5,7], 7:[5,7], 8:[5,7], 9:[1,2,6,7],\
                                            10:[1,2,6,7], 11:[1,2,6,7], 12:[4,7], 13:[5,7], 14:[]}
        # Create dictionarry of new attack stages - which stage is reached after each action taken in current stage
        # e.g., being in recon stage, taking action 1, then end up in attack stage 1 or
        # being in cyberattack stage, taking action 4, then end up in attack stage 1 (1:[[0,1], [1,4])
        self.new_stage_dict = {
            0:[0,0], 1:[[0,1],[1,4]], 2: [[2,4],[5,1]], 3: [[3,4],[9,1]], 4: [[4,4],[11,1]],
            5: [[0,2],[5,5]], 6: [[1,2],[6,5]], 7:[[7,5],[9,2]], 8: [[8,5],[10,2]], 9:[[0,3],[9,6]],
            10: [[1,3],[10,6]], 11:[[5,3],[11,6]], 12: [[10,1],[12,4]], 13: [[11,2],[13,5]],
            14:[[0,7],[1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[9,7],[10,7],[11,7],[12,7],[13,7],[14,7]]
        }
        # Determine set of contacts to which malicious agents are connected in sn
        self.mcontacts = set().union(*(self.neighbours[magent] for magent in self.malagents)) - set(self.malagents)

        while len(self.mcontacts) > 54:
            self.mcontacts.pop()
    
        # Add elements if the current size is less than the target size
        regagents_not_in_mcontacts = set(self.regagents) - self.mcontacts
        while len(self.mcontacts) < 54 and regagents_not_in_mcontacts:
            new_element = regagents_not_in_mcontacts.pop()
            self.mcontacts.add(new_element)

        # Form dictionary of contacts and corresponding actions ({contact: action})
        self.mcontacts_dict = {element: action for action, element in enumerate(self.mcontacts)}
        # Create {malagent:id} dictionary to keep actions and rewards for actions in place
        self.malagents_dict = {element: action for action, element in enumerate(self.malagents)}

    def _create_pairs_from_graph(self, actions):
        """
        Determine agent pairs from social network graph that can interact with each other
        """
        # nodes = list(self.social_network.nodes())
        nodes = self.sn_agents
        pairs = []
        # Copy the list of all nodes to remaining nodes to be able to modify it
        remaining_nodes = nodes.copy()

        # Check if sn botnet should be created
        if actions["malagent"][0] in [2,3,5,6]:
            sn_actions_dict = actions["malagent"][2] # sn_actions {'malagent12': 9, 'malagent4': 3, 'malagent8': 5} - value is action
            # Reverse the dictionary to map from action to contact {action: contact}
            action_to_contact = {v: k for k, v in self.mcontacts_dict.items()}
            # Loop over each contact in action in dict
            for malagent_name, action in sn_actions_dict.items():
                # Determine malagent id
                malagent = int(malagent_name.replace("malagent", "")) # malagent's ID
                contact = action_to_contact[action]
                if contact not in remaining_nodes: # (this check is necessary to pass the api test)
                    # If action already chosen, choose another action
                    exclusive_nodes = [node for node in remaining_nodes if node not in self.malagents]
                    contact = np.random.choice(exclusive_nodes)
                # Create a tuple of malagent and contact
                pair = (malagent, contact)
                # Append to pairs list
                pairs.append(pair)
                # Remove the paired nodes from the remaining nodes list
                remaining_nodes.remove(contact)
                remaining_nodes.remove(malagent)

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
    
    def _create_dos_affected_agents(self, actions):
        """
        Creates a list of regular agents who were affected by a denial of service attack
        retrun:     list of agents (integers)
        """
        if actions["malagent"][1] != -1:            
            # Determine how many malicious requests were blocked by the defender
            blocked_requests = actions[f"defagent{self.attacked_provider}"][0]
            n_blocked_malagents = np.sum((self.malagent_indices == True) & (blocked_requests == 1))
            # Determine how many malicious agents were allocated to a botnet
            bot_action = actions["malagent"][1] # botnet action
            bot_size = bot_action + 1 # botnet size is 1 more than the action (-1 = 0)
            # Check if botnet size is greater than the number of blocked malicious requests
            if n_blocked_malagents < bot_size:
                # Subtract the number of malicious requests that were blocked by the defender
                bot_size -= n_blocked_malagents
                # Determine regular agents who requested service from attacked service provider
                regagent_actions = {agent: values for agent, values in actions.items() if agent.startswith('regagent')}
                customers = []
                for agent_name, (provider, _) in regagent_actions.items():
                    agent = int(re.findall(r'\d+', agent_name)[0])
                    if provider == self.attacked_provider:
                        customers.append(agent)
                # NOTE: Assumption: If defender fails to block all malicious requests,
                # the number of regular agents' requests that equals the remaining botnet size is blocked.
                # Sample a random list of customers who were affected by a denial of service attack
                dos_affected_agents = random.sample(customers, k=bot_size)
            else:
                dos_affected_agents = []
        else:
            dos_affected_agents = []
        return dos_affected_agents

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
            # Determine provider's central transition matrix (from -1 to -1, from -1 to 1, from 1 to -1, from 1 to 1)
            center_matrix = [[1 - self.center_down_to_up[provider], self.center_down_to_up[provider]],\
                             [self.center_up_to_down[provider], 1 - self.center_up_to_down[provider]]]

            # Determine the next state based on the transition probabilities
            center_state = self.center[provider_name]
            next_center = random.choices([0, 1], weights=center_matrix[center_state])[0]
            self.center[provider_name] = next_center

        for provider in self.providers:
            provider_name = f"defagent{provider}"
            # Determine provider's endpoint transition matrix (from -1 to -1, from -1 to 1, from 1 to -1, from 1 to 1)
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
    
    def _determine_service_availability(self, provider_name):
        """
        Determine availability of the attacked service provider
        """
        provider = int(provider_name.replace("defagent", ""))
        # Endpoint state
        agent_data = self.endpoint
        # Define availability as equal to endpoint_state
        availability = np.array([0 if values[provider] == 0 else values[provider] for values in agent_data.values()])
        return tuple(availability)
    
    def _update_attack_stage(self, action):
        """
        Determine the next attack stage based on the taken attack stage action
        """
        # Get current attack stage
        list_observations = list(self.observations["malagent"])
        attack_stage = list_observations[-1]

        # Form a target element which includes attack stage and action 
        target_element = [attack_stage, action]

        if target_element == [0,0]:
            new_state = 0
        else:
            new_state = None
            # Find the target element from the new stage dictionary
            for key, values in self.new_stage_dict.items():
                for value in values:
                    if value == target_element:
                        # Store key as new state
                        new_state = key
                        break
                if new_state is not None:
                    break

        return new_state

    def _monitor_requests_and_sn(self, actions):
        # Initialise dictionaries
        service_requests = {f"defagent{provider}": [0] * len(self.sn_agents) for provider in self.providers}
        negative_opinions = {f"defagent{provider}": [0] * len(self.sn_agents) for provider in self.providers}

        # Populate the dictionaries based on the agents' actions data
        for agent_name, (provider, opinion) in actions.items():
            agent = int(re.findall(r'\d+', agent_name)[0])
            
            if provider != -1:
                service_requests[f"defagent{provider}"][agent] = 1
            
            if opinion in self.positive_opinions:
                sp = self.opinions_dict[opinion][1]
                negative_opinions[f"defagent{sp}"][agent] = 1 # positive opinion marked as 1
            elif opinion in self.negative_opinions:
                sp = self.opinions_dict[opinion][1]
                negative_opinions[f"defagent{sp}"][agent] = 2 # negative opinion marked as 2
            # NOTE: does not save -1 values
        
        # Combine service_requests and negative_opinions into a single dictionary
        self.defenders_observations = {f"defagent{provider}": (service_requests[f"defagent{provider}"], negative_opinions[f"defagent{provider}"]) 
                                       for provider in self.providers}

        return self.defenders_observations

    def _calculate_service_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on service and cost
        return:     float
        """
        agent = int(agent_name.replace("regagent", "")) # agent id
        provider = actions[agent_name][0] # agent's action (sp id) for this time step
        provider_name = f"defagent{provider}" # service provider's name

        # Determine provider's endpoint state for this specific agent
        endpoint_state = self.endpoint[agent_name][provider] # 1: available, 0: unavailable

        # Check if service request was made unavailable by the attacker or blocked by defender
        if agent in self.dos_agents or actions[provider_name][0][agent] == 1:
            service_reward = -1
        elif endpoint_state == 1:
            service_reward = 1
        else:
            service_reward = -1

        # Determine cost associated with the action, 
        if self.actions_taken[agent_name] is None: # agent has not taken any actions yet
            reward = service_reward
        elif self.actions_taken[agent_name] == provider: # agent stayed with save providers
            reward = service_reward + self.cost[provider]
        else: # agent changed provider
            reward = service_reward - self.cost[provider]

        # Change action to the most recent action
        self.actions_taken[agent_name] = provider

        # Save requested service state and time step for this agent
        self.service_received[agent_name][provider].append((self.timestep, service_reward))
        return reward
    
    def _calculate_feedback_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on feedback from neighbour
        return:     float
        """
        # Determine interacting agents
        agent = int(agent_name.replace("regagent", "")) # agent id
        neighbour = self._get_interaction(agent) # neighbour id
        # Determine agents' opinions
        opinion = actions[agent_name][1] # agent's opinion
        # Save currently expressed opinion
        self.opinion_expressed[agent_name] = opinion
        
        # Identify the service provider associated with the expressed opinion
        agent_opinion_value = self.opinions_dict[opinion][0] # opinion value (-1 or 1)
        agent_opinion_sp = self.opinions_dict[opinion][1] # service provider id (0,1,2)
        sp_name = f"defagent{agent_opinion_sp}" # service provider name

        # Determine feedback from defender
        # Check if defenders are present and if they decided to reply to agent's opinion 
        # NOTE: (Assumption) service provider replies positively to positive opinion and negatively to negative opinion
        if actions[sp_name][1][agent] == 1:
            defender_feedback_value = 1 # defender's opinion is always positive for their service
        else:   
            defender_feedback_value = 0

        # Calculate defender's feedback reward
        defender_feedback = agent_opinion_value * defender_feedback_value

        # Determine feedback from neighbour
        if neighbour == -1: # agent has no contact this time step
            feedback = 0
        elif neighbour in self.regagents:
            neighbour_name = f"regagent{neighbour}"
            neighbour_trust = self.observations[neighbour_name][agent_opinion_sp]  # neighbour's trust for this provider
            # Determine feedback based on dynamic trust threshold
            if neighbour_trust >= self.average_situational_trust[agent_opinion_sp]:
                feedback = 1 * agent_opinion_value # neighbour has pos opinion
            else:
                feedback = -1 * agent_opinion_value # neighbour has neg opinion
        else:
            neighbour_name = f"malagent{neighbour}"
            attack_action = actions["malagent"][0] # attacker's attack stage
            # Check if attackers in misinformation stage and agent expressed opinion on target
            if (attack_action in [2,3,5,6] and opinion == self.misinfo_opinion) or (attack_action in [2,3,5,6] and opinion == self.misinfo_alt_opinion):
                feedback = -1 * agent_opinion_value
                # attacker also has negative opinion (reinforces expressed negative opinion (-1 * -1 = 1) and decreases positive (-1 * 1 = -1))
            else:
                feedback = 0 * agent_opinion_value # NOTE: (Assumption) attacker only gives feedback when opinion is about the attacked sp

        # Give additional support for opinion that agent itself has strong direct experience with
        if actions[agent_name][0] == agent_opinion_sp:
            support = feedback + self.feedback_adj_term
        else:
            support = feedback - self.feedback_adj_term

        # Save defender's feedback value at current time step
        if defender_feedback != 0:
            self.feedback_received[agent_name][opinion].append((self.timestep, defender_feedback))
        # Save neighbour's feedback value at current time step
        if feedback != 0:
            self.feedback_received[agent_name][opinion].append((self.timestep, feedback))
        return float(defender_feedback + support)

    def _calculate_filtering_rewards(self, provider_name, actions):
        """
        Calculates filtering rewards by penalising false negatives and
        false positives and rewarding correct filtering.
        return: numpy array of shape (n_agents)
        """
        # Determine defender's state 
        requests = np.array(self.defenders_observations[provider_name][0])
        # Determine defender's actions
        blocked_requests = actions[provider_name][0]

        # Initialise filtering rewards to 0
        filtering_rewards = np.zeros(len(self.sn_agents), dtype=float)

        # Identify blocked requests
        blocked = (requests == 1) & (blocked_requests == 1)
        # Identify not blocked requests
        not_blocked = (requests == 1) & (blocked_requests == 0)
       
        # Penalise false negative: malicious request not being blocked
        filtering_rewards[self.malagent_indices & not_blocked] = -1.0
        # Penalise false positives: benign request being blocked
        filtering_rewards[self.regagent_indices & blocked] = -1.0
        # Reward correct filtering (true negatives and true positives)
        filtering_rewards[self.malagent_indices & blocked] = 1.0
        filtering_rewards[self.regagent_indices & not_blocked] = 1.0
        return filtering_rewards
    
    def _calculate_sn_answer_rewards(self, provider_name, actions):
        """
        Calculates reply rewards by penalising false negatives and false positives and
        rewarding correct replies.
        return: numpy array of shape (n_agents)
        """
        # Determine defender's state
        sn_state = np.array(self.defenders_observations[provider_name][1])
        # Determine defender's actions
        answers = actions[provider_name][1]

        # Initialise replay rewards to 0
        sn_answer_rewards = np.zeros(len(self.sn_agents), dtype=float)

        # Positive opinion is not answered
        pos_not_answered = (sn_state == 1) & (answers == 0)
        # Positive opinion is answered
        pos_answered = (sn_state == 1) & (answers == 1)
        # Negative opinion is not answered
        neg_not_answered = (sn_state == 2) & (answers == 0)
        # Negative opinion is answered
        neg_answered = (sn_state == 2) & (answers == 1)

        # False Negative (FN): a defender fails to respond to a regular agent who expressed negative opinion
        sn_answer_rewards[self.regagent_indices & neg_not_answered] = -1.0
        # False Positive for Positive Opinions (FP+): a defender replies to a regular agent who expressed positive opinion
        sn_answer_rewards[self.regagent_indices & pos_answered] = -1.0
        # False Positive (FP): a defender replies to a malicious agent who expressed negative opinion
        sn_answer_rewards[self.malagent_indices & neg_answered] = -1.0

        # True Negative for Positive Opinions (TN+): a defender ignores a regular agent who expressed positive opinion
        sn_answer_rewards[self.regagent_indices & pos_not_answered] = 1.0
        # True Negative (TN): a defender ignores a malicious agent who expressed negative opinion
        sn_answer_rewards[self.malagent_indices & neg_not_answered] = 1.0
        # True Positive (TP): a defender replies to a regular agent who expressed negative opinion
        sn_answer_rewards[self.regagent_indices & neg_answered] = 1.0

        return sn_answer_rewards

    def _calculate_attacker_rewards(self, actions):
        """
        Calculate rewards for attacker
        return: float, float, dict
        """
        # Determine attack stage and action
        attack_stage = self.observations["malagent"][-1] # current attack stage
        attack_stage_action = actions["malagent"][0] # action taken in current attack stage

        # Calculate termination reward (if agent hasn't terminated already and action is "termination")
        if attack_stage != 14 and attack_stage_action == 7: # an episode is 100 timesteps
            termination_reward = -10
        else: 
            termination_reward = 0.0

        # Calculate reconnaissance reward (if attacker in reconnaissance and action is stay in reconnaissance)
        if attack_stage == 0 and attack_stage_action == 0:
            recon_reward = 10
        else: 
            recon_reward = 0.0

        # Calculate cyberattack reward (if attacker in starts cyberattack/combined attack or stays in cyberattack/combined attack)
        if attack_stage_action in [1,3,4,6]:
            cyberattack_reward = self._calculate_cyberattack_reward(actions)
        else:
            cyberattack_reward = 0.0
        
        if attack_stage_action in [2,3,5,6]:
            # Determine connected neighbours' opinion
            misinfo_reward, misinfo_rewards = self._calculate_disinfo_reward(actions) # value, and per agent
        else:
            misinfo_reward = 0.0; misinfo_rewards = {-(i+1): 0.0 for i in range(len(self.malagents))}

        # Calculate attack_stage rewards
        attack_stage_reward = termination_reward + recon_reward + cyberattack_reward + misinfo_reward
        return attack_stage_reward, cyberattack_reward, misinfo_rewards
    
    def _calculate_cyberattack_reward(self, actions):
        """
        Calculate cyberattack reward (float)
        return: float
        """
        # Determine attacked provider (defender)
        provider_name = f"defagent{self.attacked_provider}"
        # Determine agents who requested service from attacked provider (binary array 1: request, 0: no request)
        requests = np.array(self.defenders_observations[provider_name][0])
        # Determine agents who were blocked by the attacked provider (binary array 1: blocked, 0: no block)
        blocked_requests = actions[provider_name][0]
        # Determine markov states of the targeted provider (defender) (array of 1: available, 0: unavailable)
        real_endpoint_state = np.array([values[self.attacked_provider] for values in self.endpoint.values()])
        # Change endpoint 0s to -1s while keeping 1s as 1s
        endpoint_state =  np.array([-1 if real_endpoint_state[item] == 0 else 1 for item in range(len(real_endpoint_state))])
        # Determine availability of the targeted provider (defender)
        availability = endpoint_state.copy()
        availability[blocked_requests == 1] = -1 # set positions with blocked_requests=1 to -1
        # Determine service states (considering markov model and blocking)
        service = availability.copy()
        service[requests == 0] = 0 # set positions with requests=0 to 0
        # Give reward only for regular agents who did not access service
        service[self.malagents] = 0
        # Sum together all (regular agents') negative experiences
        attack_reward = abs(np.sum(service[service == -1])) * 10.0
        # Determine cost of cyberattack (= botnet size) 
        cost = actions["malagent"][1] + 1 # original reward
        # Calculate cyberattack reward
        cyberattack_reward = attack_reward - cost
        return float(cyberattack_reward)
    
    def _calculate_disinfo_reward(self, actions):
        """
        Calculate misinformation reward
        return: float, dict
        """
        # Determine attackers actions in sn (represented as dict {agent_name: action})
        sn_actions = actions["malagent"][2]
        
        # Determine agents who received misinformation as a list of regagent names
        contacted_agents = []
        # Reverse the dictionary to map from action to contact ({action:contact})
        action_to_contact = {v: k for k, v in self.mcontacts_dict.items()}
        for _, action in sn_actions.items():
            contact = action_to_contact[action]
            contact_name = f"regagent{contact}"
            contacted_agents.append(contact_name)

        # Only deal with regular agents actions ({"regagent0": (provider, opinion), ...})
        regagent_actions = {agent_name: value for agent_name, value in actions.items() if agent_name.startswith('regagent')}
        # Initialise action rewards with zeros
        action_rewards = {action: 0.0 for _, action in sn_actions.items()}
        # Iterate over regular agents actions to get rewards
        for agent_name, (provider, opinion) in regagent_actions.items():
            # Check if contacted agent takes service from attacked provider
            if agent_name in contacted_agents:
                agent = int(agent_name.replace("regagent", "")) # contact id
                # provider_name = f'defagent{provider}' # agents service provider
                if opinion == self.misinfo_opinion  or opinion == self.misinfo_alt_opinion or \
                        (provider == self.attacked_provider and agent in self.dos_agents) or\
                            (provider == self.attacked_provider and  actions[f"defagent{provider}"][0][agent] == 1):
                    action = self.mcontacts_dict[agent] # action id corresponding to contact id
                    action_rewards[action] = 10.0
        # Consider cost of all actions (= number of malicious agents)
        cost = len(self.malagents)
        # Sum all action rewards
        sum_action_rewards = sum(action_rewards.values()) - cost
        return float(sum_action_rewards), action_rewards
    
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
        direct_exp_score = (pos_exps + 1)/(pos_exps + neg_exps + 2) # tau = (r+1)/(r+s+2)

        # Analyse witness information
        pos_wit = np.zeros(self.nProviders, dtype = float) #r
        neg_wit = np.zeros(self.nProviders, dtype = float) #s

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
        Calculate average situational trust and dynamic thresholds for each service provider,
        considering only `regagent` observations.
        Returns:
            mean_trusts (np.ndarray): Mean trust values for each provider.
            dynamic_thresholds (np.ndarray): Dynamic thresholds for each provider.
        """
        # Extract regagent observations
        regagent_observations = {
            agent: trust for agent, trust in self.observations.items() if agent.startswith("regagent")
        }

        # Initialise arrays to collect trust values for each provider
        situational_trust_values = [[] for _ in range(self.nProviders)]

        # Collect situational trust values for each provider from regagent observations
        for agent_trust in regagent_observations.values():
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
