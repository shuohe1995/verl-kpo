# KPO (Online Causal Kalman Filtering for Stable and Effective Policy Optimization)

This directory contains the **KPO** implementation: Kalman filtering is used to smooth token-level importance ratios within verl’s PPO framework, with optional PPO-style clipping.

## Directory structure

```
recipe/kpo/
├── README.md              # This file
├── main_kpo.py            # Entry point (Hydra + Ray PPO trainer)
├── core_algos.py          # KPO core (Kalman filter and policy loss)
├── config/
│   └── kpo_trainer.yaml   # KPO defaults (inherits ppo_trainer)
└── kpo_qwen3_4b.sh       # Example train/eval script for Qwen3-4B
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


1. **Training** (example):
   ```bash
   cd /path/to/verl
   bash recipe/KPO/kpo_qwen3_4b.sh
   ```

2. **Evaluation**: Pass `eval` or `evaluation` to use evaluation data and batch settings:
   ```bash
   bash recipe/KPO/kpo_qwen3_4b.sh eval
   ```


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