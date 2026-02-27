# do train
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
code_root=/mnt/bn/data-content-nlp-yg/user/youwangjie/code/aweme-rl

model_path=/mnt/bn/data-content-nlp-yg/user/youwangjie/hf_models/Qwen/Qwen2.5-7B-Instruct
train_files="['$code_root/open_verl/data/kk_logic/train.parquet']"
test_files="['$code_root/open_verl/data/kk_logic/test.parquet']"

project_name='awemerl_exp'
experiment_name='awmerl_kk'

CKPTS_DIR=$code_root/exps/${project_name}_${experiment_name}/ckpts
LOG_PATH=$code_root/exps/${project_name}_${experiment_name}/run.log
mkdir -p $CKPTS_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    2>&1 | tee -a ${LOG_PATH}