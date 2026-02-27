import json
import argparse
from pathlib import Path
from typing import List, Any

def merge_json_files(input_files: List[str], output_file: str) -> None:
    """
    合并多个 JSON 文件（每个文件是 list），输出到单个文件
    """
    merged_data = []
    
    for file in input_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {file} is not a list, skipping.")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged {len(input_files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON files (each containing a list)")
    parser.add_argument("--input", nargs='+', required=True, help="Input JSON files (e.g., 1.json 2.json ...)")
    parser.add_argument("--output", required=True, help="Output JSON file (e.g., merged.json)")
    args = parser.parse_args()
    
    merge_json_files(args.input, args.output)