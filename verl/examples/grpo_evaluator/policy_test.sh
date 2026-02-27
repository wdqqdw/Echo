#!/usr/bin/env bash
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
code_root=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments

model_path=/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/model/Qwen2.5-7b
nnodes=1

# 测试数据文件路径
test_files="[ '/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/Policy_test/test.parquet']"
project_name='awemerl_exp'
experiment_name='awmerl_policy'
CKPTS_DIR=$code_root/exps/${project_name}_${experiment_name}/ckpts
LOG_PATH=$code_root/exps/${project_name}_${experiment_name}/eval.log
mkdir -p $CKPTS_DIR

# 加载的模型权重文件路径
model_state_dict_path="/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps/awemerl_exp_awmerl_policy/ckpts/global_step_120/data.pt"
extra_state_path="/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps/awemerl_exp_awmerl_policy/ckpts/global_step_120/actor/extra_state_world_size_8_rank_0.pt"

# 自定义奖励函数文件路径和函数名
custom_reward_function_path="/opt/tiger/open_verl/verl/utils/reward_score/policy_test.py"
custom_reward_function_name="compute_f1_reward"

python3 -m verl.trainer.main_eval \
    data.path="$test_files" \
    data.response_key="response" \
    data.data_source_key="data_source" \
    data.reward_model_key="reward_model" \
    ray_init.num_cpus=8 \
    custom_reward_function.path="$custom_reward_function_path" \
    custom_reward_function.name="$custom_reward_function_name" \
    2>&1 | tee -a ${LOG_PATH}