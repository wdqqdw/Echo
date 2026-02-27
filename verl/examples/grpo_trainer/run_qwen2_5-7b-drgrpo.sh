set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
export VLLM_ATTENTION_BACKEND=XFORMERS
nnodes=1

train_files="[ '/mnt/bn/awemelm-lf-nas/users/youwangjie/code/aweme_rl/open_verl/data/math/train.parquet']"
test_files="[ '/mnt/bn/awemelm-lf-nas/users/youwangjie/code/aweme_rl/open_verl/data/math/test.parquet']"

project_name='awemerl_test'
experiment_name='qwen2_7b_drgrpo_math8k_h20'

MODEL_PATH=/mnt/bn/awemelm-lf-nas/users/youwangjie/hf_models/Qwen/Qwen2.5-7B
CKPTS_DIR=/mnt/bn/awemelm-lf-nas/users/youwangjie/code/aweme_rl/exps/${project_name}_${experiment_name}/ckpts
LOG_PATH=/mnt/bn/awemelm-lf-nas/users/youwangjie/code/aweme_rl/exps/${project_name}_${experiment_name}/run.log
mkdir -p $CKPTS_DIR

# actor_rollout_ref:
#   actor:
#     loss_agg_mode: "seq-mean-token-sum-norm" # turn off seq-dim averaging
#     use_kl_loss: False
# algorithm:
#   norm_adv_by_std_in_grpo: False # turn off standard deviation norm

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    2>&1 | tee -a ${LOG_PATH}