# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# Multi-agent CAGE Challenge 3 environment: n_hot learning agents share a policy while the remaining
# drones run the symbolic Canary (CW) protocol. Provides the revised observation space (paper Table 4).

from statistics import mean
from typing import Optional
from CanaryAgent.CanaryAgentRaw import CanaryAgent
from CanaryAgent.CanaryPZWrapper_OH7 import ObsCommsPettingZooParallelWrapper
from CybORG.Simulator.Actions.GreenActions.SendData import SendData


class NHotEnv(ObsCommsPettingZooParallelWrapper):
    def __init__(self, env, n_hot=18):
        super(NHotEnv, self).__init__(env)
        self.n_hot = n_hot
        if self.n_hot == 18:
            self.hot_agent_ids = list(range(n_hot))
        else: # Randomise which agents are 'hot' 
            self.hot_agent_ids = self.np_random.choice( list(range(self.num_drones)), size=self.n_hot, replace=False)
        
        self.hot_agent_names = [self.possible_agents[hid] for hid in self.hot_agent_ids]

        if self.n_hot < self.num_drones:
            self.heuristic_agents = {agent_id: CanaryAgent(id=agent_id[11:], np_random=self.np_random) for agent_id in self.possible_agents if agent_id not in self.hot_agent_names}
        else:
            self.heuristic_agents = {}

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> dict:
        """
        Reset env and return obs
        """
        rtn = super().reset(seed, return_info, options)

        if self.n_hot == 18:
            self.hot_agent_ids = list(range(self.n_hot))
        else: # Randomise which agents are 'hot'
            self.hot_agent_ids = self.np_random.choice( list(range(self.num_drones)), size=self.n_hot, replace=False)

        self.hot_agent_names = [self.possible_agents[hid] for hid in self.hot_agent_ids]

        if self.n_hot < self.num_drones:
            self.heuristic_agents = {agent_id: CanaryAgent(id=agent_id[11:], np_random=self.np_random) for agent_id in self.possible_agents if agent_id not in self.hot_agent_names}
        else:
            self.heuristic_agents = {}

        return {hin: rtn[hin] for hin in self.hot_agent_names}

    def step(self, hot_actions: dict):
        """
        Take the hot action, get the actions from the CanaryAgents and
        return: obs, reward, done, info for the hot_agent only.
        """

        raw_obs = {agent_id: self.env.get_observation(agent_id) for agent_id in self.possible_agents if agent_id not in self.hot_agent_names}
        actions = {}
        messages = {}

        # Get actions and messages from heuristic agents using raw_obs
        if self.n_hot < self.num_drones:
            actions = {agent_id: heuristic_agent.get_action(raw_obs[agent_id]) for agent_id, heuristic_agent in self.heuristic_agents.items()}
            messages = {agent_id: msg[1] for agent_id, msg in actions.items()}
            actions = {agent_id: act[0] for agent_id, act in actions.items()}
        
        # Get action and message for hot agents
        acts, msgs = self.select_messages(hot_actions) # Generate message only for hot agents
        for hid, han in hot_actions.items():
            actions[hid] = self.int_to_cyborg_action(hid, acts[hid])
            messages[hid] = msgs[hid]

        # for agent_id in actions.keys():
        #     print(agent_id, ' action: ', raw_obs[agent_id][2], ' nxt act: ', actions[agent_id])
        
        #print(acts.values())
        # Progress environment with all agents and messages 
        raw_obs, rews, dones, infos = self.env.parallel_step(actions, messages=messages)

        # Return obs for hot agent
        obs = {han: self.observation_change(han, raw_obs[han]) for han in self.hot_agent_names}
        self.dones.update(dones)
        self.rewards = {agent: float(sum(agent_rew.values())) for agent, agent_rew in rews.items() if agent in self.hot_agent_names}

        # Calculate the original reward by subtracting -1 for each unroutable SendData action
        cannon_rew = mean(self.rewards.values()) - (len([act for act in self.env.environment_controller.routeless_actions if type(act)==SendData]))
        infos = {agent: {'cannon_rew': cannon_rew} for agent, agent_rew in rews.items()}

        return obs, \
            {han: self.rewards[han] for han in self.hot_agent_names}, \
            {han: dones[han] for han in self.hot_agent_names}, \
            {han: infos[han] for han in self.hot_agent_names}

    def render(self):
        return self.env.render()

