sudo pip install ms-swift==3.5.2
sudo pip install deepspeed --upgrade
sudo pip install "decord" -U
sudo pip install qwen-omni-utils[decord] -U
sudo pip install accelerate

cd /path/to/Echo/ms-swift

export NPROC_PER_NODE=4
export MASTER_ADDR=localhost
export MASTER_PORT=`echo $METIS_WORKER_0_PORT | cut -d ',' -f 1`
export RANK=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/path/to/Echo/ms-swift/nccl_debug.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CHECKPOINT_SERIALIZATION=1

swift sft \
    --model "/path/to/Qwen2.5-Omni-7B" \
    --train_type full \
    --dataset '/path/to/sft_data' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-6 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps $(expr 16 / $NPROC_PER_NODE) \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /path/to/save/cold_start_model \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' 
    #    --system 'You are a helpful assistant.' \