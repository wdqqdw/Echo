#!/usr/bin/env bash
set -euxo pipefail
export VLLM_ATTENTION_BACKEND=XFORMERS
project_name='verl_dapo'
exp_name='Qwen2.5-Math-7B-DAPO-1node'

adv_estimator=grpo

kl_coef=0.0
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz *3))
train_prompt_mini_bsz=32
n_resp_per_prompt=16

use_token_level_loss=True
CODE_ROOT_PATH=/opt/tiger/open_verl

RUNTIME_ENV=${RUNTIME_ENV:-"${CODE_ROOT_PATH}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
CUR_DIR=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps
TRAIN_FILE=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/math/dapo-math-17k.parquet
TEST_FILE=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/math/aime-2024.parquet
MODEL_PATH=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/model/Qwen2.5-Math-7b
LOG_PATH=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps/${exp_name}.log
CKPTS_DIR=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps/ckpts/${exp_name}

# Algorithm
## Train
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=True

cd $CODE_ROOT_PATH
python3 -m verl.trainer.main_ppo \
   data.train_files="${TRAIN_FILE}" \
   data.val_files="${TEST_FILE}" \
   data.prompt_key=prompt \
   data.truncation='left' \
   data.max_prompt_length=${max_prompt_length} \
   data.max_response_length=${max_response_length} \
   data.train_batch_size=${train_prompt_bsz} \
   data.truncation='left' \
   actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
   actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
   actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
   actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
   algorithm.adv_estimator=${adv_estimator} \
   algorithm.kl_ctrl.kl_coef=${kl_coef} \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
   actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
   actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
   actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
   actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
   actor_rollout_ref.model.path="${MODEL_PATH}" \
   +actor_rollout_ref.model.override_config.attention_dropout=0. \
   +actor_rollout_ref.model.override_config.embd_pdrop=0. \
   +actor_rollout_ref.model.override_config.resid_pdrop=0. \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.optim.lr=1e-6 \
   actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
   actor_rollout_ref.actor.optim.weight_decay=0.1 \
   actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
   actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
   actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.actor.grad_clip=1.0 \
   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   actor_rollout_ref.rollout.enable_chunked_prefill=True \
   actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
   actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
   actor_rollout_ref.rollout.val_kwargs.top_p=1.0\
   actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
   actor_rollout_ref.rollout.val_kwargs.n=1 \
   actor_rollout_ref.rollout.val_kwargs.do_sample=True \
   actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
   actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
   actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
   actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
   trainer.logger=['console','wandb'] \
   trainer.project_name="${project_name}" \
   trainer.experiment_name="${exp_name}" \
   trainer.n_gpus_per_node=8 \
   trainer.nnodes="${NNODES}" \
   trainer.test_freq=10 \
   trainer.save_freq=10 \
   trainer.total_epochs=30 \
   trainer.default_local_dir="${CKPTS_DIR}" \
   trainer.resume_mode=disable \
   2>&1 | tee  ${LOG_PATH}