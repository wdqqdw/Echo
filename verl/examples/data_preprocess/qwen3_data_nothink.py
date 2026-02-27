import argparse
import os
import pandas as pd
from tqdm import tqdm

def process_parquet(input_path, output_path):
    """
    处理单个Parquet文件，向prompt字段的content添加/no_think指令
    """
    print(f"Reading Parquet file from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print("Processing prompts...")
    
    # 确保prompt列存在
    if 'prompt' not in df.columns:
        print("Error: 'prompt' column not found in the Parquet file.")
        return
    
    modified_prompts = []
    for prompt in tqdm(df['prompt']):
        try:
            # 处理可能的numpy数组或其他序列类型
            if hasattr(prompt, '__iter__') and len(prompt) > 0:
                first_message = prompt[0]
                # 检查是否是类字典对象
                if hasattr(first_message, 'get'):
                    content = first_message.get('content', '')
                    
                    # 添加/no_think指令（确保前面有空格）
                    if not content.endswith(' /no_think'):
                        if content and not content.endswith(' '):
                            content += ' '
                        content += '/no_think'
                        
                        # 更新content
                        first_message['content'] = content
                    
                    modified_prompts.append(prompt)
                    continue
        except Exception as e:
            print(f"Error processing prompt: {e}")
        
        # 如果格式不符合预期，保留原始值
        print(f"Unexpected format for prompt: {prompt[:100]}...")
        modified_prompts.append(prompt)
    
    # 更新DataFrame中的prompt列
    df['prompt'] = modified_prompts

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存修改后的Parquet文件
    print(f"Saving modified Parquet file to: {output_path}")
    df.to_parquet(output_path)
    
    print(f"Successfully processed {len(df)} records.")
    
    # 验证处理结果
    if len(df) > 0:
        first_prompt = df['prompt'].iloc[0]
        try:
            if hasattr(first_prompt, '__iter__') and len(first_prompt) > 0:
                first_message = first_prompt[0]
                if hasattr(first_message, 'get'):
                    first_content = first_message.get('content', '')
                    print(f"First prompt content ends with '/no_think': {first_content.endswith('/no_think')}")
                    print(f"First prompt example: {first_content[:100]}...")
        except Exception as e:
            print(f"Error verifying prompt: {e}")

def main():
    parser = argparse.ArgumentParser(description='Add /no_think instruction to prompts in Parquet files')
    parser.add_argument('--input', required=True, help='Input Parquet file path')
    parser.add_argument('--output', required=True, help='Output Parquet file path')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    # 处理单个Parquet文件
    process_parquet(args.input, args.output)

if __name__ == "__main__":
    main()

# 示例python3 /opt/tiger/open_verl/examples/data_preprocess/qwen3_data_nothink.py --input /mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/Policy/test.parquet --output /mnt/bn/chuwei-nas-hl/users/jinhaopeng/experiments/open_verl/data/verl/Policy_nothink/test.parquet