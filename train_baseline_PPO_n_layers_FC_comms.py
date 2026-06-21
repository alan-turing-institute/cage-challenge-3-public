# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# Train PPO with the explicit 16-bit communications channel (standard obs + messages; model M2).

import os
import argparse

# ML imports
import torch
import gym
import ray
from ray import tune, air
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.registry import get_algorithm_class
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


# CybORG imports
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from MAPPODivRewParallelPettingZooEnv import MAPPODivRewParallelPettingZooEnv
from CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import AgentCommsPettingZooParallelWrapper

NUM_DRONES = 18
AGENT_LABELS = ["blue_agent_"+str(i) for i in range(NUM_DRONES)]
POLICY_LABELS = ["blue_agent"]                                      # For single shared policy 
# POLICY_LABELS = ["blue_agent_"+str(i) for i in range(NUM_DRONES)] # For one-policy-per-agent

# RLlib Algorithm config
config_baseline_PPO={                    
    "env": "CC3",
    "env_config": {},       # Config dict passed to the custom env's constructor.
    "num_gpus": torch.cuda.device_count(),
    "num_workers": 16,       # Parallelize environment roll-outs.
    "framework": "torch",
    "eager_tracing": True,  # This makes debugging harder but speeds execution.
    "explore": True,
    "rollout_fragment_length": 50,
    "multiagent": {         # MARL config
        "policies": POLICY_LABELS,
        "policies_to_train": POLICY_LABELS,
        "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "blue_agent"),
        "count_steps_by": "agent_steps" # https://github.com/ray-project/ray/issues/12372
        # Use `**kwargs: agent_id' For one-policy-per-agent.
    },
    "model": {
            "custom_model": "CC3_nFC_Model",
            "vf_share_layers": False,
    },
}

# Define custom CAGE Challenge 3 environment
def env_creator_CC3(env_config: dict):
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg, environment='sim')
    env = MAPPODivRewParallelPettingZooEnv(AgentCommsPettingZooParallelWrapper(env=cyborg))
    return env

# Try import torch
torch, nn = try_import_torch()

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        model_config['fcnet_hiddens'] = [256, 2]    # FC hidden-layer widths (as used to train the released checkpoints; do not change)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

ModelCatalog.register_custom_model("CC3_nFC_Model", TorchModel) 

if __name__ == "__main__":

    ray.init()
    register_env(name="CC3", env_creator=env_creator_CC3)

    # Stop training config
    stop = {
        "training_iteration":  25000000,   # Number of iterations to train
        "timesteps_total":     25000000,   # Number of timesteps to train
        "episode_reward_mean": 0,      # Reward at which to stop training
    }

    # Tune model using PPO with custom Torch model
    print("Training policy for {} timesteps.".format(stop["timesteps_total"]))
    tuner = tune.Tuner(
        "PPO",
        param_space=config_baseline_PPO,
        run_config=air.RunConfig(
            name="CC3_PPO_comms_baseline",
            local_dir="logs/ray/PPO0/",
            stop=stop,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, 
                num_to_keep = 3,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_at_end=True
            ),
        ),
    )
    results = tuner.fit()

    print("Training completed.")

    # Write the logdir to pointer file.
    latest_logdir = results.get_best_result().log_dir.as_uri()
    logdir_ptr_fname = 'checkpoint_baseline_PPO.txt'
    with open(logdir_ptr_fname, 'w') as f:
        f.write(latest_logdir)
    print('Best checkpoint: {}'.format(latest_logdir))
    print('written to {}'.format(logdir_ptr_fname))

    ray.shutdown()