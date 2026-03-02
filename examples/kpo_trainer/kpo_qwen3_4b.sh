set -xeuo pipefail

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    train_path=$HOME/verl/data/math/train.parquet
    test_path=$HOME/verl/data/math/test.parquet
    train_batch_size=32
    val_batch_size=64
    val_before_train=True
    n_resp_per_prompt=8
    val_n_resp_per_prompt=16
else
    echo "Running in training mode"
    train_path=$HOME/verl/data/math/train.parquet
    test_path=$HOME/verl/data/math/test_sampled.parquet
    train_batch_size=32
    val_batch_size=110
    val_before_train=false
    n_resp_per_prompt=8
    val_n_resp_per_prompt=1
fi

# can make training faster, depends on your infrastructure
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Set how many GPUs we actually have on this node.
export GPUS_PER_NODE=2

SLURM_JOB_NUM_NODES=1
NNODES=${SLURM_JOB_NUM_NODES}
export NNODES

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY}"       # set in env, e.g. export WANDB_API_KEY=your_key
export WANDB_DIR="${WANDB_DIR}"               # set in env, e.g. export WANDB_DIR=/path/to/checkpoints
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"  # set in env, e.g. export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="${HF_HOME}"                   # set in env, e.g. export HF_HOME=/path/to/hf_cache


echo "Using $NNODES nodes for training..."

# ------------------------------------- Setup xp params ---------------------------------------
project_name='RL-KPO'
adv_estimator=grpo
loss_mode=kpo 
is_clip=true  # true: PPO-style clip; false: unclipped KPO
kalman_q=1e-6   # KPO Kalman process noise (Q)
kalman_r=1      # KPO Kalman observation noise (R)
loss_agg_mode="seq-mean-token-mean"
MODEL_PATH=Qwen/Qwen3-4B  # Qwen/Qwen3-4B
rollout_engine=vllm
rollout_mode=async
return_raw_chat="True"
if [ "$rollout_engine" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
gpu_memory_utilization=0.6
reward_manager=dapo
adv_estimator=grpo
shuffle_dataset=true
first_time_dataset_prep=true # prepare dataset

test_freq=10
save_freq=100
total_epochs=2

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1 0.0003
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1 0.0004
ppo_mini_batch_size=8 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory

max_prompt_length=$((1024))
max_response_length=$((1024 * 4))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Paths and namings
SFT_MODEL=$(basename $MODEL_PATH)
exp_name="${loss_mode}-epslow-${clip_ratio_low}-epshigh-${clip_ratio_high}-${SFT_MODEL}-now-noadaclip-causal-q1e-6-r1"
CKPTS_DIR=/mnt/raid/data/shuohe/checkpoints/${loss_mode}/${exp_name}
#ollout_data_dir=/mnt/raid/data/shuohe/rollout_data/${exp_name}

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=true
gen_tp=1
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.

# ------------------------------------- train/val data preparation ---------------------------------------

python examples/data_preprocess/kpo_math.py --local_dir ${HOME}/verl/data/math


# set the paths
train_files="['$train_path']"
test_files="['$test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    +actor_rollout_ref.actor.is_clip=${is_clip} \
    +actor_rollout_ref.actor.kalman_Q=${kalman_q} \
    +actor_rollout_ref.actor.kalman_R=${kalman_r} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.return_raw_chat=${return_raw_chat} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_resp_per_prompt}  \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \