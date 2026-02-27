import os
import json
import random
import pandas as pd
from datasets import Dataset

# 文件路径
input_file = '/mnt/bn/chuwei-nas-hl/users/jinhaopeng/data/Policy/for_doubao/aweme_0318_for_doubao_mixed_only_moderation_random_round1.jsonl'
output_dir = '/mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/Policy/qwen3_sft'

os.makedirs(output_dir, exist_ok=True)
# 加载数据
data = []
with open(input_file, 'r') as f:
    for line in f:
        try:
            obj = json.loads(line)
            question = obj['messages'][0]['content']
            answer = obj['messages'][1]['content']
            
            # 调整数据结构，使prompt和response成为字典
            data.append({
                "question": question,  # 保留顶级字段
                "answer": answer,      
                "data_source": "policy_logic",
                "prompt": {"question": question},  # prompt是包含question键的字典
                "response": {"answer": answer},    # response是包含answer键的字典
                "ability": "policy_review",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {}
            })
        except (KeyError, IndexError, json.JSONDecodeError):
            continue

print(f"成功加载 {len(data)} 条数据")

# 打乱数据
random.shuffle(data)

# 划分训练集和测试集 (90% 训练, 10% 测试)
split_ratio = 0.9
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
test_data = data[split_index:]

print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

# 为训练集和测试集添加index和split信息
for i, item in enumerate(train_data):
    item["extra_info"] = {"index": i, "split": "train"}
for i, item in enumerate(test_data):
    item["extra_info"] = {"index": i, "split": "test"}

# 创建数据集对象
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

# 保存为parquet文件
train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(output_dir, 'test.parquet'))

print(f"数据处理完成，训练集和测试集已保存到 {output_dir}")

# 从指定的test.parquet文件中加载一条数据
test_parquet_path = os.path.join(output_dir, 'test.parquet')
try:
    table = pd.read_parquet(test_parquet_path)
    if not table.empty:
        first_case = table.iloc[0].to_dict()
        print("从test.parquet加载的一条case:")
        print(first_case)
    else:
        print("test.parquet文件为空。")
except Exception as e:
    print(f"读取测试数据时出错: {e}")