# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# PettingZoo wrapper and helpers for the Canary (CW) protocol (message scheduling and action encoding).

from typing import Optional, Tuple
import numpy as np
from gym import spaces

# CryBORG
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from CybORG.Simulator.Actions import Sleep, Action
from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.RetakeControl import RetakeControl
from CybORG.Simulator.Actions.ConcreteActions.RemoveOtherSessions import RemoveOtherSessions
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTraffic
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTraffic

from CanaryAgent.Commons import pad_canary_message, unpad_canary_message, cyborg_action_to_int

DEBUG = False


class ObsCommsPettingZooParallelWrapper(PettingZooParallelWrapper):
    """Communicates part of the previous observation to other agents"""

    def __init__(self, env):
        super(ObsCommsPettingZooParallelWrapper, self).__init__(env)
        self.np_random = env.np_random
        self.num_drones = len(self.ip_addresses)
        self.message_space = self.env.get_message_space(self.possible_agents[0]).n
        self.msg_len = self.num_drones * self.message_space

        self._observation_spaces = {agent_name: spaces.MultiDiscrete(
            [3] +                                               # last action success?
            [self.action_space(self.possible_agents[0]).n] +    # last action number
            [2 for i in range(self.num_drones)] + 
            [2] + 
            [3 for i in range(self.num_drones)] + 
            [101, 101] + 
            (self.num_drones - 1) * [self.num_drones, 101, 101, 2] +
            [2 for i in range(self.num_drones)] +                               # to_fix list
            [2 for _ in range(self.message_space)]*self.num_drones)             # Messages
                for agent_name in self.possible_agents}

        self.time = {agent_name:-1 for agent_name in self.possible_agents}
        self.to_fix = {agent_name:{} for agent_name in self.possible_agents}
        for agent_name in self.possible_agents:
            self.to_fix[agent_name] = {int(agent_name.split('_')[-1]):0 for agent_name in self.possible_agents}
        self.position = {agent_name:[-1, -1] for agent_name in self.possible_agents}
        self.neighbours = {agent_name:{} for agent_name in self.possible_agents}
        self.last_action = {agent_name:[self.action_space(agent_name).n-1, self.action_space(agent_name).n-1] for agent_name in self.possible_agents}

    def parse_message(self, message: list, agent_name: str):
        """Parse message for append to obs space"""
        new_message = message
        while len(new_message) < self.num_drones:
            new_message.append([0 for _ in range(self.env.get_message_space(agent_name).n)])
        
        return np.array(new_message).flatten()

    def select_messages(self, action):
        """
            Selects the messages to send to other agents.
            :param action: dict specifying an action for each agent
        """
        for agent, act in action.items():
            if type(act) == int:
                self.last_action[agent] = [act] + self.last_action[agent][:1]   
            else:   
                self.last_action[agent] = [cyborg_action_to_int(act)] + self.last_action[agent][:1]  
        
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
            if (self.time[agent] > 1) and host_name not in obs:
                # If this drone is infected blow whistle on self
                msg[agent] = pad_canary_message(canary=host_id, overheard=None, whistle=host_id)
                if DEBUG:# and agent == 'blue_agent_12':
                    print('agent {} is infected, sending {}'.format(agent, unpad_canary_message(msg[agent])))
                continue
            
            # If we aren't infected...
            #print("{} not infected".format(host_name))
            try:
                new_pos = obs[host_name]['System info']['position']
                new_pos[0] = round(new_pos[0])
                new_pos[1] = round(new_pos[1])
            except KeyError:
                pass

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

                    for m in self.np_random.choice(cleaned_messages, len(cleaned_messages), replace=False):
                        # Randomly sample from the received messages
                        m_canary, m_overhead, m_whistle = unpad_canary_message(m)
                        # print(m)
                        # print(m_canary, m_overhead, m_whistle)

                        if m_canary >= 0: 
                            self.neighbours[agent][m_canary] = self.time[agent]

                        if host_id != m_canary and m_overhead == 1:
                            # Notified by other drone about infected drone
                            canary = m_canary
                            m_warn = m_whistle
                            #print(m_canary, m_overhead, m_whistle)
                            if m_warn >= 0 and m_warn <18 and m_warn not in self.to_fix[agent]:
                                if DEBUG: # and agent == 'blue_agent_12':
                                    print("( Drone", host_id, ") Notified about drone", m_warn, "by drone", m_canary)
                                self.to_fix[agent][m_warn] = 1

                for n_id in self.neighbours[agent].keys():

                    if self.neighbours[agent][n_id] <= self.time[agent]-1:
                        # We stopped getting messages from this drone
                        # Infected drone detected - inform others (vs *I* am infected, i.e., overheard=1)
                        if DEBUG and agent == 'blue_agent_12':
                            print("( Drone", host_id, ") Trojan detected on drone", n_id)
                        self.to_fix[agent][n_id] = 1
                        msg[agent] = pad_canary_message(canary=host_id, overheard=1, whistle=n_id)

                    elif self.neighbours[agent][n_id] == self.time[agent]:
                        # drone not infected, but is on to_fix
                        if self.to_fix[agent][n_id] == 1:
                            # remove from to_fix
                            if DEBUG and agent == 'blue_agent_12':
                                print("( Drone", host_id, ") thinks ", n_id, " is now clean")
                            self.to_fix[agent][n_id] = 0
            
            # Default canary/heartbeat message
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
        # assuming that the final value in the agent name indicates which drone that agent is on
        if 'agent' in agent:
            self.agent_host_map = {agent_name: f'drone_{agent_name.split("_")[-1]}' for agent_name in self.possible_agents}
            # get all ip_addresses
            self.ip_addresses = list(self.env.get_ip_map().values())
            num_drones = len(self.ip_addresses)
            obs_length = int(1 +    # last action success
                             1 +    # last action
                             num_drones + 1 + num_drones + 2 + 
                             (num_drones - 1) * (2 + 1 + 1) + 
                             num_drones +   # modify obs to include to_fix
                             #num_drones +   # and last action performed on each drone
                             self.msg_len)
            new_obs = np.zeros(obs_length, dtype=int)
            if obs is not None:
                own_host_name = self.agent_host_map[agent]
                # obs_length = success + own_drone(block Ips + processes + network conns) + other_drones_including_own(IPs + session_ + pos)
                # element location --> [0, 1,...,num_drones, 1+num_drones, 2+num_drones, ..., 2+2*num_drones, 3+2*num_drones, 4+2*num_drones,...,4+4*num_drones]
                index = 0
                # success
                new_obs[index] = obs['success'].value - 1
                index += 1
                # last action
                new_obs[index] = self.last_action[agent][-1]
                index += 1

                if agent in self.env.active_agents:
                    # Add blocked IPs
                    for i, ip in enumerate(self.ip_addresses):
                        new_obs[index + i] = 1 if ip in [blocked_ip for interface in
                                                         obs[own_host_name]['Interface'] if
                                                         'blocked_ips' in interface for blocked_ip in interface['blocked_ips']] else 0
                    index += len(self.ip_addresses)

                    # add flagged malicious processes
                    new_obs[index] = 1 if 'Processes' in obs[own_host_name] else 0
                    index += 1
                    # add flagged messages
                    for i, ip in enumerate(self.ip_addresses):
                        new_obs[index + i] = 1 if ip in [network_conn['remote_address']
                                                         for interface in obs[own_host_name]['Interface']
                                                         if 'NetworkConnections' in interface
                                                         for network_conn in interface['NetworkConnections']] \
                            else 0
                    index += len(self.ip_addresses)

                    pos = obs[own_host_name]['System info'].get('position', (0, 0))
                    new_obs[index] = max(int(pos[0]), 0)
                    new_obs[index + 1] = max(int(pos[1]), 0)
                    index += 2
                    ip_host_map = {ip: host for host, ip in self.env.get_ip_map().items()}
                    # add information of other drones
                    for i, ip in enumerate(self.ip_addresses):
                        hostname = ip_host_map[ip]
                        if hostname != own_host_name:
                            new_obs[index] = i
                            index += 1
                            # add position of drone
                            if hostname in obs:
                                pos = obs[hostname]['System info'].get('position', (0, 0))
                                new_obs[index] = max(int(pos[0]), 0)
                                new_obs[index + 1] = max(int(pos[1]), 0)
                                index += 2
                                # add session to drone
                                new_obs[index] = 1 if 'Sessions' in obs[hostname] else 0
                                index += 1
                            else:
                                new_obs[index] = 0
                                new_obs[index + 1] = 0
                                new_obs[index + 2] = 0
                                index += 3
                    # Add to_fix
                    for to_fix in self.to_fix[agent]:
                        new_obs[index] = self.to_fix[agent][to_fix]
                        index += 1
                    
                    if DEBUG and agent == 'blue_agent_12':
                        print(agent, new_obs[index-18:index])
                        pass
                    # Add messages
                    msg = self.parse_message(obs['message'] if 'message' in obs else [], agent)
                    if len(msg) > 0:
                        for j in range(len(msg)):
                            new_obs[index+j] = msg[j]
                        index += len(msg)
                # update data of other drones
                # try:
                assert self._observation_spaces[agent].contains(
                    new_obs), f'Observation \n{new_obs}\n is not contained within Observation Space \n{self._observation_spaces[agent]}\n for agent {agent}'
                # except AssertionError:
                #     breakpoint()
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
        self.last_action = {agent_name:[self.action_space(agent_name).n-1, self.action_space(agent_name).n-1] for agent_name in self.possible_agents}

        return rtn