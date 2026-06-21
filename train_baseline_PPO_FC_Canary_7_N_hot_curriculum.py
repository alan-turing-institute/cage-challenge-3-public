# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# Train the Mixed Opera agent: an expert curriculum that gradually increases the number of learning
# agents amongst symbolic Canary/CW experts (paper Section 8.2; model M4).

import os
import argparse
from math import isnan
from statistics import mean, stdev
from datetime import datetime
import shutil

# ML imports
import gym
import ray
import torch
from ray import tune, air
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
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
from NHotEnv7 import NHotEnv

N = int(os.environ.get("N_HOT_AGENTS", 14))   # learning agents at this curriculum stage (paper grows 1 -> 14)


LOCAL_DIR = "logs/cc3/N{}_hot_curriculum/".format(N)

log_config = {
    "stop": {
        "training_iteration":  1500000,   # Number of iterations to train
        "timesteps_total":     1500000,   # Number of timesteps to train
        "episode_reward_mean": 0,      # Reward at which to stop training
    }
}                                 # For single shared policy 

# RLlib Algorithm config
config_baseline_PPO={                       
    "env": "CC3",
    "env_config": {},       # Config dict passed to the custom env's constructor.
    "num_gpus": torch.cuda.device_count(),
    "num_workers": 31,       # Parallelize environment roll-outs.
    "framework": "torch",
    "eager_tracing": True,  # This makes debugging harder but speeds execution.
    "explore": True,
    "rollout_fragment_length": 25,
    "model": {
            "custom_model": "CC3_nFC_Model",
            "vf_share_layers": False,
    },
    "logger_config": log_config,
}

def custom_logger_creator(config):
    "Custom logger defines log_dir for unrolled training loops"

    name = "CC3_MAPPO_curriculum"
    timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = "{}/{}/{}/".format(LOCAL_DIR, name, timestr)

    return UnifiedLogger(config, logdir=logdir)

def env_creator_CC3(env_config: dict): 
    """Return custom CAGE Challenge 3 environment for PettingZoo"""
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg, environment='sim')
    env = MAPPODivRewParallelPettingZooEnv(NHotEnv(env=cyborg, n_hot=N))
    #env = ParallelPettingZooEnv(ObsCommsPettingZooParallelWrapper(env=cyborg))
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
    # The curriculum bootstraps stage N from the checkpoint trained at stage N-1 (see README).
    # Point CW_RESUME_CHECKPOINT at that checkpoint directory; empty trains this stage from scratch.
    checkpoint = os.environ.get("CW_RESUME_CHECKPOINT", "") or None
    if checkpoint and not os.path.exists(checkpoint):
        print("No checkpoint at {} - training from scratch.".format(checkpoint)); checkpoint = None
    if checkpoint is None:
        print("Training this curriculum stage from scratch.")

    register_env(name="CC3", env_creator=env_creator_CC3)

    

    # Tune model using PPO with custom Torch model
    print("Training policy for {} timesteps.".format(log_config["stop"]["timesteps_total"]))

    checkpoints = []
    ma10 = [-9000.0]*10
    best_ma_10_checkpoints = []
    
    if checkpoint:
        # Restore from checkpoint
        trainer = PPOTrainer(config=config_baseline_PPO, env="CC3", logger_creator=custom_logger_creator)
        trainer.restore(checkpoint)
    else:
        # Start from scratch
        trainer = PPOTrainer(config=config_baseline_PPO, env="CC3", logger_creator=custom_logger_creator)

    for i in range(log_config["stop"]["training_iteration"]):
        result = trainer.train()
        if i % 1 == 0:  # adjust the frequency of saving as needed
            reward = result['episode_reward_mean']
            
            # Add the checkpoint to the list
            if not isnan(reward):
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint, "with reward", reward)
                checkpoints.append((checkpoint, reward))
                ma10 = [reward] + ma10[:9] #FILO
                ma10_ = mean(ma10)
                ma_10_ckpt = trainer.save(trainer.logdir+'ma10/')
                best_ma_10_checkpoints.append((ma_10_ckpt, ma10_))

            # Once there are more than 4 checkpoints, delete the one with the lowest reward
            if len(checkpoints) > 4:

                best_ma_10_checkpoints.sort(key=lambda x: x[1], reverse=True) # Sort high to low
                oldest_ma10, _ =  best_ma_10_checkpoints.pop(-1) # remove lowest
                shutil.rmtree(oldest_ma10)

                # Sort the list by reward in descending order
                checkpoints.sort(key=lambda x: x[1], reverse=True)
                oldest_checkpoint, _ = checkpoints.pop(-1)
                shutil.rmtree(oldest_checkpoint)


        if result['timesteps_total'] >= log_config["stop"]["timesteps_total"]:
            break
    
    # Save the final checkpoint
    checkpoint = trainer.save()
    print("Final checkpoint saved at", checkpoint)
    print("Training completed.")

    ray.shutdown()