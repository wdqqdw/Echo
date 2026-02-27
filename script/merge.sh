cd /path/to/Echo/verl

python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/verl/checkpoint \
    --target_dir /path/to/merged_model

cp /path/to/Qwen2.5-Omni-7B/spk_dict.pt /path/to/merged_model
