#bash /mnt/bn/wdq-base1/data/ALMs/think_with_audio/setup.sh

cd /path/to/Echo
n_gpu=8
experiment_name=demo
checkpoint=/path/to/merged_model
prompt=inference/grounding_1_0715.txt
experiment_dir=output/$experiment_name
# 检查目录是否存在
if [ ! -d "$experiment_dir" ]; then
    mkdir -p "$experiment_dir"
fi

benchmarks=("MMAR" "MMAU-mini" "MMAU")

for benchmark in ${benchmarks[@]}; do
    echo "Processing benchmark: $benchmark"
    
    output_dir=$experiment_dir/$benchmark
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi

    n_gpu_1=`expr $n_gpu - 1`

    # 存储所有后台进程的 PID
    pids=()
    for i in $(seq 0 $n_gpu_1); do
        if test -f "$output_dir/$i.json"; then
            echo "$output_dir/$i.json exists, skip."
            continue
        fi
        CUDA_VISIBLE_DEVICES=$i \
        nohup python inference/inference_multiturn.py \
            --checkpoint $checkpoint \
            --prompt $prompt \
            --benchmark $benchmark \
            --output_file $output_dir/$i.json \
            --rank $i \
            --total_rank $n_gpu > \
            "$output_dir/infer_$i.log" 2>&1 &
        # 记录当前后台进程的 PID
        pids+=($!)
    done
    # 等待所有后台进程完成
    wait "${pids[@]}"
    echo "checkpoint: $checkpoint inference on: $benchmark has finished."
    echo "Results has been saved to: $output_dir"

    input_files=()
    for i in $(seq 0 $n_gpu_1); do
        input_files+=($output_dir/$i.json)
    done
    python inference/merge_json.py \
        --input "${input_files[@]}" \
        --output $output_dir/merged.json

    if [ "$benchmark" = "MMAR" ]; then
        python /mnt/bn/wdq-base1/data/ALMs/datasets/MMAR/code/evaluation.py \
            --input $output_dir/merged.json > $output_dir/evaluation.txt
    elif [ "$benchmark" = "MMAU-mini" ]; then
        python /mnt/bn/wdq-base1/data/ALMs/datasets/MMAU/evaluation.py \
            --input $output_dir/merged.json > $output_dir/evaluation.txt
    fi

done

#pkill -f "inference"