#!/bin/bash
set -x

# 基础路径配置
BASE_EXP_DIR="/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/exps/awemerl_exp_sft_Qwen3_1_7b_2048_left"
MODEL_SAVE_PATH="${BASE_EXP_DIR}/checkpoints"
LOG_DIR="${BASE_EXP_DIR}/logs"
CONFIG_FILE="/opt/tiger/open_verl/verl/trainer/config/sft_trainer.yaml"  # 指定配置文件路径

# 定义模型路径和数据路径
LOCAL_MODEL_PATH="/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/model/Qwen3-1.7B"
DATA_PATH="/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/Policy/qwen3_debug"

# 创建目录结构
mkdir -p "${MODEL_SAVE_PATH}"
mkdir -p "${LOG_DIR}"

# 记录启动时间和主机信息
START_TIME=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)
LOG_FILE="${LOG_DIR}/train_${START_TIME}_${HOSTNAME}.log"

# 检查本地模型路径是否存在
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "错误: 本地模型路径不存在: $LOCAL_MODEL_PATH"
    exit 1
fi

# 检查数据路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据文件是否存在
TRAIN_FILE="${DATA_PATH}/train.parquet"
VAL_FILE="${DATA_PATH}/test.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "错误: 验证数据文件不存在: $VAL_FILE"
    exit 1
fi

echo "数据文件检查通过:"
echo "  训练数据: $TRAIN_FILE"
echo "  验证数据: $VAL_FILE"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <nproc_per_node> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
shift  # 移除第一个参数，保留其他参数


torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.fsdp_sft_trainer_local \
    --config-name sft_trainer \
    --config-path /opt/tiger/open_verl/verl/trainer/config \
    model.partial_pretrain="${LOCAL_MODEL_PATH}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    trainer.default_local_dir="${MODEL_SAVE_PATH}" \
    trainer.project_name=Awemerl-Qwen3 \
    trainer.experiment_name=full-finetune \
    'trainer.logger=["console", "wandb"]' \
    trainer.default_hdfs_dir=null \
    model.fsdp_config.wrap_policy.name=transposed \
    model.fsdp_config.wrap_policy.disable=false \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.05 \
    model.enable_gradient_checkpointing=true \
    "$@" 2>&1 | tee -a "${LOG_FILE}"

# 输出日志路径信息
echo "训练日志已保存到: ${LOG_FILE}"
echo "模型检查点将保存到: ${MODEL_SAVE_PATH}"