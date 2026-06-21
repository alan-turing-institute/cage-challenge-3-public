# Validation harness for the Mixed Opera (curriculum MAPPO) model.
# Loads the N14 curriculum checkpoint and evaluates it with `n_hot` neural agents
# (sharing the learned policy) + (18 - n_hot) symbolic CanaryAgents, under the TRUE
# reward, reproducing the paper's Figure-5 curve. Chris Hicks paper reproduction, 2026.
import argparse, importlib, warnings, time
warnings.filterwarnings("ignore")
from statistics import mean, stdev

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPO

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from NHotEnv7 import NHotEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-hot", type=int, default=7)       # number of neural (learning) agents; rest are CW experts
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    # Reuse the N-hot train module for the custom-model registration + PPO config (single default_policy).
    M = importlib.import_module("train_baseline_PPO_FC_Canary_7_N_hot")
    config = dict(M.config_baseline_PPO)
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config.pop("logger_config", None)

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=3, logging_level="ERROR")
    register_env(name="CC3", env_creator=M.env_creator_CC3)

    print(f"Restoring curriculum checkpoint:\n  {args.checkpoint}", flush=True)
    algo = PPO(config=config)
    algo.restore(args.checkpoint)
    print("Restore OK. Policy ids:", list(algo.workers.local_worker().policy_map.keys()), flush=True)

    sg = DroneSwarmScenarioGenerator()
    # one env instance, reset per episode (matches author's eval); random seed -> proper episode variance
    env = NHotEnv(env=CybORG(scenario_generator=sg, environment="sim"), n_hot=args.n_hot)

    returns = []
    t0 = time.time()
    for i in range(args.episodes):
        obs = env.reset()
        r = []
        for j in range(args.steps):
            actions = algo.compute_actions(obs, explore=True)   # default_policy, shared across the n_hot agents
            obs, rew, done, info = env.step(actions)
            vals = list(rew.values())
            if vals:
                r.append(mean(vals))
            if done and all(done.values()):
                break
        returns.append(sum(r))
        rm = mean(returns); sd = stdev(returns) if len(returns) > 1 else 0.0
        print(f"  ep {i+1:3d}/{args.episodes} (n_hot={args.n_hot}): {sum(r):9.1f}   running mean {rm:9.1f}  std {sd:7.1f}", flush=True)

    el = time.time() - t0
    print("\n" + "=" * 70)
    print(f"MODEL        = Mixed Opera (N14 curriculum)")
    print(f"CHECKPOINT   = {args.checkpoint}")
    print(f"N_HOT        = {args.n_hot} neural + {18-args.n_hot} CanaryAgent(CW) experts")
    print(f"EPISODES     = {len(returns)} x {args.steps} steps")
    print(f"MEAN REWARD  = {mean(returns):.1f}")
    print(f"STD DEV      = {stdev(returns) if len(returns)>1 else 0:.1f}")
    print(f"(wall {el:.0f}s, {el/max(1,len(returns)):.2f}s/ep)")
    print("=" * 70)
    ray.shutdown()

if __name__ == "__main__":
    main()
