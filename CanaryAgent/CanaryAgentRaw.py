# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# The Canaries-and-Whistles (CW) expert agent: the symbolic blue-team protocol of Algorithm 1.

from inspect import signature
from typing import Union
import time, math, random

from gym import Space
import numpy as np

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTraffic, AllowTraffic
from CybORG.Simulator.Actions.ConcreteActions.RemoveOtherSessions import RemoveOtherSessions
from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.RetakeControl import RetakeControl
from CybORG.Simulator.Actions import Sleep

from CanaryAgent.CanaryPZWrapper import pad_canary_message, unpad_canary_message


DEBUG = False

class CanaryAgent(BaseAgent):
    """Runs the Heuristic Canary Protocol"""

    def __init__(self, id=None, np_random = None):
        super().__init__(np_random)
        self.np_random = np_random
        self.action_params = None
        self.id = int(id)
        self.agent_name = 'blue_agent_' + str(id)
        self.drone_name = 'drone_' + str(id)
        self.time = -1
        self.ip_list = {}
        self.neighbours = {}
        self.to_fix = {}
        self.position = [-1, -1]
        self.infected = False


    def train(self, results):
        pass

    def get_action(self, observation: dict):
        self.time += 1

        if self.time == 0: 
            for k, v in observation.items():
                if (k not in ['success', 'message']):
                    self.ip_list[k] = [v['Interface'][0]['IP Address']][0]
            #print("Collected ips:", self.ip_list)

        # Heartbeat message
        # bnr = np.binary_repr(int(self.id)+1, width=16)
        # msg_heartbeat = np.array(list(bnr), dtype=int)
        msg_heartbeat = pad_canary_message(self.id, overheard=None, whistle=None)
        msg = msg_heartbeat

        # Infected
        if self.time > 1 and self.drone_name not in observation:
            if DEBUG:
                print("RAW: ( Drone", self.id, ") I am infected")
            self.infected = True

            # bnr_warn = np.binary_repr(int(self.id + 1), width=6)
            # msg_warn = np.array(list(bnr_warn), dtype=int)
            # msg_warn = np.pad(msg_warn, (0, 16 - len(msg_warn)), 'constant', constant_values=(0, 0))
            msg_warn = pad_canary_message(self.id, overheard=None, whistle=self.id)
            action = RemoveOtherSessions(agent=self.agent_name, session=0)
            if DEBUG:
                print("RAW: ( Drone", self.id, ") self cleaning")
            return action, msg_warn
            #return Sleep(), msg_heartbeat

        # If we made it here, things look good.
        if self.infected == True:
            if DEBUG:
                print("RAW: ( Drone", self.id, ") Disinfected!!!")
            self.infected = False

            # This may not be necessary
            action = RemoveOtherSessions(agent=self.agent_name, session=0)
            if DEBUG:
                print("RAW: ( Drone", self.id, ") self cleaning")
            return action, msg_heartbeat

        # Not infected
        try:
            new_pos = observation[self.drone_name]['System info']['position']
            new_pos[0] = round(new_pos[0])
            new_pos[1] = round(new_pos[1])
        except KeyError:
            pass

        if (self.time > 2) and (new_pos[0] != self.position[0] and new_pos[1] != self.position[1]):
            #print("We moved! From", self.position, "to", new_pos, "(id:", self.agent_name,")")
            self.position = new_pos
            self.neighbours = {} # New neighbours
            #self.to_fix = {} # New to fix
        else:
            if 'message' in observation:
                #print("id", self.id, "Messages:", len(observation['message']), observation['message'])
                for m in self.np_random.choice(observation['message'], len(observation['message']), replace=False):
                    m_canary, m_overheard, m_whistle = unpad_canary_message(m)
                    #m_heartbeat = m[10:]
                    #id_heartbeat = int(''.join(map(lambda m_heartbeat: str(int(m_heartbeat)), m_heartbeat)), 2)-1
                    id_heartbeat = m_canary
                    # print("( Drone", self.id, ") Got heartbeat from", id_heartbeat)
                    if id_heartbeat >= 0 and id_heartbeat <18:
                        self.neighbours[id_heartbeat] = self.time

                    if self.id != id_heartbeat and m_overheard == 1:
                        # Notified by others about infected drone
                        # m_warn = m[:6]
                        #id_warn = int(''.join(map(lambda m_warn: str(int(m_warn)), m_warn)), 2) - 1 # ID - 1
                        id_warn = m_whistle
                        # print("warning FROM others int:", id_warn)
                        # print("warning FROM others binary:", m_warn)
                        if (id_warn>=0 and id_warn<18) and id_warn not in self.to_fix:
                            # print(m_warn)
                            if DEBUG:
                                print("RAW: ( Drone", self.id, ") Notified about drone", id_warn, "by drone", id_heartbeat)
                            self.to_fix[id_warn] = "block_overheard"

            for id in self.neighbours:
                if self.neighbours[id] <= self.time-1:
                    if DEBUG:
                        print("RAW: ( Drone", self.id, ") Trojan detected on drone", id)
                    self.to_fix[id] = "block_detected"

                    # Tell others about this infected drone
                    # NOTICE THAT id+1
                    #bnr_warn = np.binary_repr(int(id+1), width=6)
                    # print("warning others int:", id+1)
                    # print("warning others binary:", bnr_warn)
                    # msg_warn = np.array(list(bnr_warn), dtype=int)
                    # msg_warn = np.pad(msg_warn, (0, 16-len(msg_warn)), 'constant', constant_values=(0, 0))
                    # msg_warn[9] = 1 # Flag for warning
                    #msg = msg_heartbeat + msg_warn
                    msg = pad_canary_message(canary=self.id, overheard=1, whistle=id)
                    # print("msg_warn", msg_warn)
                    # print("msg_heartbeat", msg_heartbeat)
                    # print("msg", msg)


            if len(self.to_fix) > 0:
                id = self.np_random.choice(list(self.to_fix.keys()),1, replace=False)[0]

                # first Block
                if self.to_fix[id] == "block_overheard" or self.to_fix[id] == "block_detected":
                    self.to_fix[id] = "retake"
                    if self.np_random.random() < 0.225: # Fast retake
                        if DEBUG:
                            print("RAW: ( Drone", self.id, ") Fast retaking drone", id)
                        return RetakeControl(agent=self.agent_name, session=0, ip_address=self.ip_list["drone_" + str(id)]), msg
                    else:
                        if DEBUG:
                            print("RAW: ( Drone", self.id, ") Blocking traffic from drone", id)
                        return BlockTraffic(agent=self.agent_name, session=0, ip_address=self.ip_list["drone_"+str(id)]), msg

                # then Retake
                elif self.to_fix[id] == "retake":
                    self.to_fix[id] = "allow"
                    if DEBUG:
                        print("RAW: ( Drone", self.id, ") Retaking drone", id)
                    return RetakeControl(agent=self.agent_name, session=0, ip_address=self.ip_list["drone_"+str(id)]), msg

                # then Allow
                elif self.to_fix[id] == "allow":
                    del self.to_fix[id]
                    if DEBUG:
                        print("RAW: ( Drone", self.id, ") Allowing traffic from drone", id)
                    return AllowTraffic(agent=self.agent_name, session=0, ip_address=self.ip_list["drone_"+str(id)]), msg

        # Default: Perform self-cleaning and send heartbeat
        return RemoveOtherSessions(agent=self.agent_name, session=0), msg_heartbeat

    def end_episode(self):
        pass

    def set_initial_values(self, action_space):
        if type(action_space) is dict:
            self.action_params = {action_class: signature(action_class).parameters for action_class in action_space['action'].keys()}

    def reset(self):
        self.time = -1
        self.ip_list = {}
        self.neighbours = {}
        self.to_fix = {}
        self.position = [-1, -1]
        self.infected = False