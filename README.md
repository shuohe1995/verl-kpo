<div align="center">

# KPO: Online Causal Kalman Filtering for Stable and Effective Policy Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2602.10609-b31b1b.svg)](https://arxiv.org/abs/2602.10609)
[![HF Daily Paper](https://img.shields.io/badge/HF-Daily%20Paper-FFD21E.svg)](https://huggingface.co/papers/2602.10609)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/shuohe1995/verl-kpo.svg)](https://github.com/shuohe1995/verl-kpo)

</div>

**`KPO`** is designed for stable and effective policy optimization of large language models via reinforcement learning (RL). It uses online causal Kalman filtering to smooth token-level importance ratios in the PPO/GRPO pipeline, reducing variance and improving training stability.

This implementation features plug-in integration with [verl](https://github.com/volcengine/verl) (loss mode `kpo`), optional PPO-style clipping, configurable Kalman noise (Q/R), and flexible loss aggregation, making it well suited for math reasoning and other RLHF/RL from human feedback workloads.

## Directory structure

```
examples/
├── kpo_trainer/
│   └── kpo_qwen3_4b.sh   # Train/eval script for Qwen3-4B (calls verl.trainer.main_ppo)
└── data_preprocess/
    └── kpo_math.py       # Data preprocessing for math KPO

verl/
├── trainer/ppo/
│   └── core_algos.py     # KPO core (Kalman filter + policy loss, registered as loss_mode "kpo")
└── utils/reward_score/
    ├── __init__.py       # default_compute_score for data_source "math_kpo"
    └── kpo_math_reward.py
```

## Algorithm overview

- **Token observation**: Use the token-level log ratio `z_t = log π(a_t|s_t) - log π_old(a_t|s_t)` as the observation.
- **Kalman filter**: Apply a 1D causal Kalman filter to `z_t` to obtain a smoothed estimate `x_hat_t`, using the non-steady gain sequence `K_1..K_T` (computed by `kalman_gains_nonsteady_1d`).
- **Importance ratio**: Build a Kalman-filtered importance ratio `kf_importance_ratio = exp(log π - log π_detach + x_hat_detach)` and use it in place of the raw token-level ratio.
- **Policy loss**:
  - **Clipped** (`is_clip=true`): As in PPO, clip `kf_importance_ratio` to `[1 - clip_ratio_low, 1 + clip_ratio_high]`, then take the max of the two objectives multiplied by advantages.
  - **Unclipped** (`is_clip=false`): Use `-advantages * kf_importance_ratio` directly.

## Main hyperparameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `loss_mode` | Policy loss type; use `kpo` | `kpo` |
| `is_clip` | Whether to use PPO-style clipping | `true` / `false` |
| `kalman_Q` | Kalman process noise (Q) | `1e-6` |
| `kalman_R` | Kalman observation noise (R) | `1` |
| `clip_ratio_low` | Clipping lower bound (when `is_clip=true`) | `0.0003` |
| `clip_ratio_high` | Clipping upper bound (when `is_clip=true`) | `0.0004` |
| `loss_agg_mode` | How to aggregate the loss | `seq-mean-token-mean` |

## Quick start

1. **Conda env install**: refer to https://verl.readthedocs.io/en/latest/start/install.html

2. **Environment variables** (near the top of `kpo_qwen3_4b.sh`): Set the following in your environment before running; the script passes them through as-is:
   | Variable | Description | Example |
   |----------|-------------|---------|
   | `WANDB_API_KEY` | Weights & Biases API key for logging training/eval runs | `export WANDB_API_KEY=your_key` |
   | `WANDB_DIR` | Directory for W&B local cache and checkpoints | `export WANDB_DIR=/path/to/checkpoints` |
   | `CUDA_VISIBLE_DEVICES` | Comma-separated list of visible GPUs | `export CUDA_VISIBLE_DEVICES=0,1` |
   | `HF_HOME` | Hugging Face cache directory (models, tokenizers, etc.) | `export HF_HOME=/path/to/hf_cache` |

3. **Training** (from repo root):
   ```bash
   cd /path/to/verl-kpo
   bash examples/kpo_trainer/kpo_qwen3_4b.sh
   ```

4. **Evaluation**: Pass `eval` or `evaluation` to use evaluation data and batch settings:
   ```bash
   bash examples/kpo_trainer/kpo_qwen3_4b.sh eval
   ```


**Weights & Biases** — Training and evaluation runs are logged to W&B. Click the badges to open the dashboards:

- [![W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge.svg)](https://api.wandb.ai/links/hs827083890-nanyang-technological-university-singapore/wcvvdkyz) **Training**
- [![W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge.svg)](https://api.wandb.ai/links/hs827083890-nanyang-technological-university-singapore/3oo5r65x) **Evaluation**

## Acknowledgement

This codebase is built upon [veRL](https://github.com/volcengine/verl). The KPO policy loss and Kalman filtering are integrated into verl's PPO/GRPO trainer and core algorithms.

The math reward is adapted from [Dr. MAS](https://github.com/langfengQ/DrMAS) and DAPO-style math evaluation [DAPO](https://github.com/verl-project/verl-recipe/tree/3490a22a0a3adeb7e4787fe70b1060b642efbae4/dapo), [DeepScaleR](https://github.com/rllm-org/rllm).

We extend our gratitude to the authors and contributors of these projects for their valuable work.

## Citation

```bibtex
@misc{he2026onlinecausalkalmanfiltering,
      title={Online Causal Kalman Filtering for Stable and Effective Policy Optimization}, 
      author={Shuo He and Lang Feng and Xin Cheng and Lei Feng and Bo An},
      year={2026},
      eprint={2602.10609},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10609}, 
}
```
