# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# RLlib ParallelPettingZoo wrapper that splits the shared team reward across agents (for MAPPO training).

from ray.rllib.env import ParallelPettingZooEnv

class MAPPODivRewParallelPettingZooEnv(ParallelPettingZooEnv):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.num_agents = len(obs.keys())
        return obs

    def step(self, action_dict):
        next_obs, rewards, dones, infos = super().step(action_dict)
        
        # Divide rewards by the number of agents
        rewards = {agent: rewards[agent] / self.num_agents for agent in rewards}
        
        return next_obs, rewards, dones, infos
