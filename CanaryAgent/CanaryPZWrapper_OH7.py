# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# PettingZoo wrapper providing the revised (Table 4) observation space shared by the learning and CW agents.

from typing import Optional, Tuple
import numpy as np
from gym import spaces

# CryBORG
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper

from CanaryAgent.Commons import pad_canary_message, unpad_canary_message, cyborg_action_to_int

DEBUG = False

class ObsCommsPettingZooParallelWrapper(PettingZooParallelWrapper):
    """
        Communicates to other agents
    """

    def __init__(self, env):
        super(ObsCommsPettingZooParallelWrapper, self).__init__(env)
        self.np_random = env.np_random
        self.num_drones = len(self.ip_addresses)
        self.message_space = self.env.get_message_space(self.possible_agents[0]).n
        self.msg_len = self.num_drones * self.message_space

        self._observation_spaces = {agent_name: spaces.MultiDiscrete(
            [self.num_drones] +                               # The ID of this drone
            [3] +                                             # last action success?
            [self.action_space(self.possible_agents[0]).n] +  # last action number (this drone)
            [3] * self.num_drones +                           # last action numbers (other drones)
            [2] +                                             # Malicious process flagged
            [2 for i in range(self.num_drones)] +             # Blocklist
            [2 for i in range(self.num_drones)])              # to_fix list
            #[2 for _ in range(self.message_space)] * self.num_drones) 
                for agent_name in self.possible_agents}

        self.time = {agent_name:-1 for agent_name in self.possible_agents}
        self.to_fix = {agent_name:{} for agent_name in self.possible_agents}
        for agent_name in self.possible_agents:
            self.to_fix[agent_name] = {int(agent_name.split('_')[-1]):0 for agent_name in self.possible_agents} # init to-fix
        self.position = {agent_name:[-1, -1] for agent_name in self.possible_agents}
        self.neighbours = {agent_name:{} for agent_name in self.possible_agents}
        self.last_action = {agent_name: [self.action_space(agent_name).n-1, self.action_space(agent_name).n-1] for agent_name in self.possible_agents} # Initialise last actions to Sleep
        self.last_actions = {agent_name: [2]*len(self.possible_agents) for agent_name in self.possible_agents} # Last known action for each drone (for each drone)

    def parse_message(self, message: list, agent_name: str):
        """Parse messages for append to obs space"""
        new_message = message
        while len(new_message) < self.num_drones:
            new_message.append([0 for _ in range(self.env.get_message_space(agent_name).n)])
        
        return np.array(new_message).flatten()

    def select_messages(self, action):
        """
            Selects the messages to send to other agents.
            Note: Called *before* parallel_step in the env i.e., these actions haven't been taken yet
            :param action: dict specifying an action for each agent
        """
        for agent, act in action.items():
            if type(act) == np.int32:
                self.last_action[agent] = [act] + self.last_action[agent][:1]
            else:   
                self.last_action[agent] = [cyborg_action_to_int(act)] + self.last_action[agent][:1]
            
            drone_id, a = self._action_to_agent_action(self.last_action[agent][1]) # Using the 'last' action
            
            if drone_id != None: # If action is remote then keep track of last performed
                self.last_actions[agent][drone_id] = a
        
        msg = {}
        for agent in action.keys():
            obs = self.env.get_observation(agent)
            self.time[agent] += 1
            #msg[agent] = [0 for i in range(self.env.get_message_space(agent).n)]
            
            # Canary
            canary = int(agent.split('_')[-1])
            host_name = self.agent_host_map[agent]
            host_id = int(host_name.split('_')[-1])
            
            # Default canary/heartbeat message
            msg[agent] = pad_canary_message(canary=host_id, overheard=None, whistle=None)

            # Infected
            if host_name not in obs:
                # If this drone is infected blow whistle on self
                msg[agent] = pad_canary_message(canary=host_id, overheard=None, whistle=host_id)
                if DEBUG:# and agent == 'blue_agent_12':
                    print('I ({}) am infected, sending {}'.format(host_name, msg[agent]))
                continue
            
            # If we aren't infected...
            try:
                new_pos = obs[host_name]['System info']['position']
                new_pos[0] = round(new_pos[0])
                new_pos[1] = round(new_pos[1])
            except KeyError:
                pass
            
            whistle = False
            if (self.time[agent] > 2) and (new_pos[0] != self.position[agent][0] and
                                           new_pos[1] != self.position[agent][1]):
                
                # If we've moved, reset neighbours and to_fix list
                self.position[agent] = new_pos
                self.neighbours[agent] = {}
                #self.to_fix[agent] = {int(agent.split('_')[-1]):0 for agent in self.possible_agents}

            else:
                if 'message' in obs:
                    # Clean messages to remove the ones that are all zeros
                    cleaned_messages = [m for m in obs['message'] if np.any(m)]

                    #for m in self.np_random.sample(cleaned_messages, len(cleaned_messages)):
                    for m in self.np_random.choice(cleaned_messages, len(cleaned_messages), replace=False):
                        # Randomly sample from the received messages
                        m_canary, m_overheard, m_whistle = unpad_canary_message(m)
                        # print(m_canary, m_overhead, m_whistle)

                        if m_canary >= 0 and m_canary < 18: 
                            self.neighbours[agent][m_canary] = self.time[agent]

                        if host_id != m_canary and m_overheard == 1 and m_whistle != None:
                            # Notified by other drone about infected drone
                            canary = m_canary
                            m_warn = m_whistle
                            #print(m_canary, m_overhead, m_whistle)
                            if m_warn not in self.to_fix[agent]:
                                if DEBUG: # and agent == 'blue_agent_12':
                                    print("( Drone", host_id, ") Notified about drone", m_warn, "by drone", m_canary)
                                self.to_fix[agent][m_warn] = 1

                for n_id in self.neighbours[agent].keys():

                    if self.neighbours[agent][n_id] == self.time[agent]-1:
                        # We stopped getting messages from this drone
                        # Infected drone detected - inform others (vs *I* am infected, i.e., overheard=1)
                        if DEBUG: #and agent == 'blue_agent_12':
                            print("( Drone", host_id, ") Trojan detected on drone", n_id)
                        self.to_fix[agent][n_id] = 1
                        msg[agent] = pad_canary_message(canary=host_id, overheard=1, whistle=n_id)
                        whistle = True

                    elif self.neighbours[agent][n_id] == self.time[agent]:
                        # drone not infected, but is on to_fix
                        if self.to_fix[agent][n_id] == 1:
                            # remove from to_fix
                            if DEBUG: # and agent == 'blue_agent_12':
                                print("( Drone", host_id, ") thinks ", n_id, " is now clean")
                            self.to_fix[agent][n_id] = 0
            
            # Default canary/heartbeat message
            if not whistle:
                msg[agent] = pad_canary_message(canary=host_id, overheard=None, whistle=None)

        return action, {agent: msg[agent] for agent, act in action.items()}

    def observation_change(self, agent: str, obs: dict):
        """
        Initialises the observation space for the agent (if undefined) or modifies the observation space (if defined)

        Parameters:
            agent -> str

        OG_obs -> None/np.array
            None if undefined
            np.array if defined
        """
        if 'agent' in agent:

            obs_length = int(1+                  # Drone ID
                             1 +                 # last action success
                             1 +                 # last action 
                             self.num_drones +   # last action(s)
                             1 +                 # malicious process detect 
                             self.num_drones +   # blacklist
                             self.num_drones)    # to_fix list
                             #self.msg_len)
            
            new_obs = np.zeros(obs_length, dtype=int)
            
            if obs is not None:

                own_host_name = self.agent_host_map[agent]
                index = 0
                # This Drone ID
                new_obs[index] = int(own_host_name[6:])
                index += 1
                # This drone last action success
                new_obs[index] = obs['success'].value - 1
                index += 1
                # last action
                #print(self.last_action[agent][1:], type(self.last_action[agent][1:]))
                new_obs[index] = self.last_action[agent][1]
                index += 1
                # last known actions
                for drone_id in self.last_actions[agent]:
                    new_obs[index] = drone_id
                    index += 1

                #print('Drone {}: {}'.format(int(own_host_name[6:]), new_obs[:index]))

                # Malicious local process detected
                host_name = self.agent_host_map[agent]
                new_obs[index] = 1 if ((self.time[agent] > 1) and host_name not in obs) else 0
                index += 1

                if agent in self.env.active_agents:
                    
                    # Add blocked IPs
                    for i, ip in enumerate(self.ip_addresses):
                        new_obs[index + i] = 1 if ip in [blocked_ip for interface in
                                                         obs[own_host_name]['Interface'] if
                                                         'blocked_ips' in interface for blocked_ip in interface['blocked_ips']] else 0
                    index += len(self.ip_addresses)

                    # to_fix list
                    for to_fix in self.to_fix[agent]:
                        new_obs[index] = self.to_fix[agent][to_fix]
                        index += 1

                    # messages
                    # msg = self.parse_message(obs['message'] if 'message' in obs else [], agent)
                    # if len(msg) > 0:
                    #     for j in range(len(msg)):
                    #         new_obs[index + j] = msg[j]
                    #     index += len(msg)

            return new_obs

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> dict:
        rtn = super().reset(seed, return_info, options)
        
        # reset all variables
        self.time = {agent_name:-1 for agent_name in self.possible_agents}
        self.to_fix = {agent_name:{} for agent_name in self.possible_agents}
        for agent_name in self.possible_agents:
            self.to_fix[agent_name] = {int(agent_name.split('_')[-1]):0 for agent_name in self.possible_agents}
        self.position = {agent_name:[-1, -1] for agent_name in self.possible_agents}
        self.neighbours = {agent_name:{} for agent_name in self.possible_agents}
        self.last_action = {agent_name: [self.action_space(agent_name).n-1, self.action_space(agent_name).n-1] for agent_name in self.possible_agents}
        self.last_actions = {agent_name: [2]*len(self.possible_agents) for agent_name in self.possible_agents} 

        return rtn
    
    def _action_to_agent_action(self, action: int):
        """
            Converts a CC3 action to a type of action on a specific drone
            Drones: 0-17
            Actions: 
                0: RetakeControl
                1: BlockTraffic
                2: AllowTraffic
        """
        if action >= 0 and action < 18: # RetakeControl
            drone_id = action
            act = 0
            return drone_id, act
        
        elif action >= 19 and action < 37: # BlockTraffic
            drone_id = action-19
            act = 1
            return drone_id, act
        
        elif action >= 37 and action < 55: # AllowTraffic
            drone_id = action-37
            act = 2
            return drone_id, act
        
        else: ## >= 55 (i.e., sleep)
            return None, 0