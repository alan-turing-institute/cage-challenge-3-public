# Canaries and Whistles: Resilient Drone Communication Networks with (or without) Deep Reinforcement Learning

Code and trained models for our AISec '23 paper on autonomous cyber defence in the third
CAGE Challenge (the CybORG drone-swarm scenario). This repository is sufficient to both *evaluate* the released models and *train* them from scratch.

Paper: <https://doi.org/10.48550/arXiv.2312.04940>

## The models

The blue team defends an 18-drone swarm whose firmware is compromised; defenders are scored
over 1000 episodes of up to 500 steps (the maximum possible score is 0). The paper's heuristic
baselines (Sleep, RemoveBlueDrone, …) and the **Canaries-and-Whistles (CW) expert** (Algorithm 1,
score −1577.7) are hand-coded and have no learned weights — the CW expert lives in `CanaryAgent/`.

The four **learning-based** policies are released here under `models/`:

| Model | Description | Paper score | Re-validated\* | Directory |
|------:|-------------|:-----------:|:--------------:|-----------|
| **M1** | PPO baseline, standard observation space (paper Table 2) | −7617.8 | −7777.7 | `models/m1_ppo_baseline/` |
| **M2** | PPO with the explicit 16-bit communications channel | −6745.3 | −6752.7 | `models/m2_ppo_comms/` |
| **M3** | PPO, improved observation space (paper Table 4), all 18 agents | −6884.4 | −6960.6 | `models/m3_ppo_improved_obs/` |
| **M4** | Mixed Opera: expert curriculum mixing CW experts with learning agents | −1487.9 | −1503.5 | `models/m4_mixed_opera/` |

\* *Re-validated* = mean episodic return over **1000 episodes** (matching the paper's protocol) from re-running each released model checkpoint under the standard reward (Table 1 in the paper); each is within ~1 standard deviation of the results reported in the paper (per-episode std ≈ 600–980). Evaluation samples actions stochastically (`explore=True`), as in the original agent code, which is slightly conservative relative to greedy action selection. Every policy is a single shared fully-connected network (`fcnet_hiddens=[256, 2]`). M1/M2 use the standard CybORG observation space (action spaces `Discrete(56)` and `Discrete(896)` respectively); M3/M4 use the revised observation space (size 58, action `Discrete(56)`). The M4 figure is at 7 learning agents – the best point of the paper's Figure 5.

## Setup

> **Git LFS:** the model weights under `models/` are stored with [Git LFS](https://git-lfs.com).
> Install Git LFS and run `git lfs install` **before** cloning so the checkpoints download
> automatically; if you cloned already, run `git lfs pull` to fetch them.

Requires conda (or a Python 3.10 virtual environment).

```bash
conda env create -f environment.yml
conda activate cage-challenge-3
```

This installs Ray/RLlib 2.2.0, PyTorch 1.13.1, and the CAGE Challenge 3 CybORG environment
(pinned commit, via `requirements.txt`). `setuptools` is pinned to 65.5.1 because Ray 2.2.0
imports `pkg_resources`, which newer setuptools removed. A CPU is sufficient for evaluation;
each model trained in ~3.5 h on a single 24-core machine.

## Reproduce the results — evaluation

Each command loads a released checkpoint and reports the mean / standard-deviation episode
return under the true reward (use a smaller `--episodes` for a quick check):

```bash
# M1 — PPO baseline
python validate_paper_models.py --variant baseline --checkpoint models/m1_ppo_baseline/      --episodes 1000
# M2 — PPO + explicit comms
python validate_paper_models.py --variant comms    --checkpoint models/m2_ppo_comms/          --episodes 1000
# M3 — improved-obs PPO (all 18 agents neural)
python validate_opera.py        --checkpoint models/m3_ppo_improved_obs/ --n-hot 18 --episodes 1000
# M4 — Mixed Opera (7 learning agents + 11 CW experts): the best point of Figure 5
python validate_opera.py        --checkpoint models/m4_mixed_opera/      --n-hot 7  --episodes 1000
```

Sweeping `validate_opera.py --checkpoint models/m4_mixed_opera/ --n-hot {1..18}` reproduces the
paper's Figure 5 (average score vs number of learning agents; the rest run the CW protocol).

## Reproduce the results — training from scratch

```bash
python train_baseline_PPO_n_layers_FC.py                          # M1
python train_baseline_PPO_n_layers_FC_comms.py                    # M2
N_HOT_AGENTS=18 python train_baseline_PPO_FC_Canary_7_N_hot.py    # M3 (all-neural, from scratch)
```

**M4 (Mixed Opera)** uses an *expert curriculum*: start with one learning agent amongst CW
experts, then grow the number of learning agents, resuming each stage from the previous one:

```bash
N_HOT_AGENTS=1 python train_baseline_PPO_FC_Canary_7_N_hot_curriculum.py
# then for N = 2, 3, …, 14, resume from the previous stage's best checkpoint:
N_HOT_AGENTS=2 CW_RESUME_CHECKPOINT=<path/to/N1/checkpoint_dir> python train_baseline_PPO_FC_Canary_7_N_hot_curriculum.py
# … up to N = 14.
```

Checkpoints are written under `logs/` (git-ignored). Training is stochastic, so re-trained
models will differ slightly from the released ones.

## Repository layout

```
models/                                       the four released policies (M1–M4)
validate_paper_models.py                      evaluate M1 / M2 under the true reward
validate_opera.py                             evaluate M3 / M4 (n_hot learners + CW experts)
train_baseline_PPO_n_layers_FC.py             train M1 (standard obs)
train_baseline_PPO_n_layers_FC_comms.py       train M2 (standard obs + comms channel)
train_baseline_PPO_FC_Canary_7_N_hot.py       train N learning agents amongst CW experts (N=18 -> M3)
train_baseline_PPO_FC_Canary_7_N_hot_curriculum.py   train the Mixed Opera curriculum (M4)
NHotEnv7.py                                   multi-agent env (revised obs, Table 4) mixing learners + CW experts
MAPPODivRewParallelPettingZooEnv.py           RLlib wrapper splitting the team reward across agents
CanaryAgent/                                  the symbolic Canaries-and-Whistles (CW) expert + obs wrappers
environment.yml, requirements.txt             the software stack
```

## The environment

The scenario is CybORG's DroneSwarm (CAGE Challenge 3): 18 drones with firmware-level malware, a
green team generating communications demand, and an offensive red team. The defensive blue team
maximises communications bandwidth despite continual adversarial interference. The environment is
a pinned fork of the official CAGE Challenge 3 (<https://github.com/cage-challenge/cage-challenge-3>),
installed via `requirements.txt`.

## Citation

If you use this code or the models, please cite:

```bibtex
@inproceedings{10.1145/3605764.3623986,
  author    = {Hicks, Chris and Mavroudis, Vasilios and Foley, Myles and Davies, Thomas and Highnam, Kate and Watson, Tim},
  title     = {Canaries and Whistles: Resilient Drone Communication Networks with (or without) Deep Reinforcement Learning},
  year      = {2023},
  publisher = {Association for Computing Machinery},
  url       = {https://doi.org/10.48550/arXiv.2312.04940},
  doi       = {10.48550/arXiv.2312.04940},
  booktitle = {Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security},
  pages     = {91--101},
  numpages  = {11},
  series    = {AISec '23}
}
```

## Acknowledgements

This research was funded by the Defence Science and Technology Laboratory (Dstl), an executive
agency of the UK Ministry of Defence, under the Autonomous Resilient Cyber Defence (ARCD) programme.

## License

MIT — see [LICENSE](LICENSE). The CybORG environment is a separate dependency with its own licence.
