# Validation harness: load a trained CAGE-Challenge-3 checkpoint and evaluate it
# under the TRUE (Table-1) reward, reproducing the paper's "average over N episodes
# of up to 500 steps" protocol. Chris Hicks paper reproduction, 2026.
import argparse, importlib, warnings, time, sys
warnings.filterwarnings("ignore")
from statistics import mean, stdev

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPO

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import AgentCommsPettingZooParallelWrapper

# variant -> (train module providing config_baseline_PPO + env_creator_CC3, eval wrapper giving TRUE reward)
VARIANTS = {
    "baseline": ("train_baseline_PPO_n_layers_FC",       PettingZooParallelWrapper),       # Table 2 obs, all-18 shared policy
    "comms":    ("train_baseline_PPO_n_layers_FC_comms", AgentCommsPettingZooParallelWrapper),  # Table 2 + comms
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--explore", default="true")
    args = ap.parse_args()
    explore = args.explore.lower() == "true"

    modname, EvalWrapper = VARIANTS[args.variant]
    mod = importlib.import_module(modname)   # side effects: registers CC3_nFC_Model custom model
    config = dict(mod.config_baseline_PPO)
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["explore"] = explore

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=3, logging_level="ERROR")
    register_env(name="CC3", env_creator=mod.env_creator_CC3)

    print(f"Building PPO and restoring checkpoint:\n  {args.checkpoint}", flush=True)
    algo = PPO(config=config)
    algo.restore(args.checkpoint)
    print("Restore OK. Policy ids:", list(algo.workers.local_worker().policy_map.keys()), flush=True)

    sg = DroneSwarmScenarioGenerator()
    def make_env():
        return EvalWrapper(env=CybORG(scenario_generator=sg, environment="sim"))
    env = make_env()

    returns = []
    t0 = time.time()
    for i in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        r = []
        for j in range(args.steps):
            actions = algo.compute_actions(obs, policy_id="blue_agent", explore=explore)
            out = env.step(actions)
            if len(out) == 5:
                obs, rew, term, trunc, info = out
                done = {a: bool(term.get(a, False)) or bool(trunc.get(a, False)) for a in term}
            else:
                obs, rew, done, info = out
            vals = list(rew.values())
            if vals:
                r.append(mean(vals))
            if done and all(done.values()):
                break
        returns.append(sum(r))
        rm = mean(returns)
        sd = stdev(returns) if len(returns) > 1 else 0.0
        print(f"  ep {i+1:3d}/{args.episodes}: {sum(r):9.1f}   running mean {rm:9.1f}  std {sd:7.1f}", flush=True)

    el = time.time() - t0
    print("\n" + "=" * 70)
    print(f"VARIANT      = {args.variant}")
    print(f"CHECKPOINT   = {args.checkpoint}")
    print(f"EPISODES     = {len(returns)} x {args.steps} steps   explore={explore}")
    print(f"MEAN REWARD  = {mean(returns):.1f}")
    print(f"STD DEV      = {stdev(returns) if len(returns)>1 else 0:.1f}")
    print(f"(wall {el:.0f}s, {el/max(1,len(returns)):.2f}s/ep)")
    print("=" * 70)
    ray.shutdown()

if __name__ == "__main__":
    main()
