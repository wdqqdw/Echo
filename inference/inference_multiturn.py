import argparse
import os
import re
import json
import copy
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

DATASET_REGISTER = {
    "MMAR": {
        "audio_dir": "/path/to/MMAR",
        "qa_file": "/path/to/MMAR/MMAR-meta.json",
        "audio_key": "audio_path",
        "num_samples": 1000
    },
    "MMAU-mini": {
        "audio_dir": "/path/to/MMAU",
        "qa_file": "/path/to/MMAU/mmau-test-mini.json",
        "audio_key": "audio_id",
        "num_samples": 1000
    },
    "MMAU": {
        "audio_dir": "/path/to/MMAU",
        "qa_file": "/path/to/MMAU/mmau-test.json",
        "audio_key": "audio_id",
        "num_samples": 9000
    },
}

def create_prompt(row: Dict[str, Any], args) -> List[Dict[str, Any]]:
    """Create conversation prompt from a row of data"""
    with open(args.prompt, "r") as f:
        prompt = f.read()
    
    if "choice_key" in DATASET_REGISTER[args.benchmark]:
        choice_key = DATASET_REGISTER[args.benchmark]["choice_key"]
    else:
        choice_key = "choices"
    
    if "question_key" in DATASET_REGISTER[args.benchmark]:
        question_key = DATASET_REGISTER[args.benchmark]["question_key"]
    else:
        question_key = "question"

    if "[QUESTION]" in prompt:
        prompt = prompt.replace("[QUESTION]", row[question_key])
    if "[CHOICES]" in prompt:
        choices = row[choice_key]
        prompt = prompt.replace("[CHOICES]", str(choices))
    
    audio_dir = DATASET_REGISTER[args.benchmark]["audio_dir"]
    audio_key = DATASET_REGISTER[args.benchmark]["audio_key"]

    if row[audio_key].startswith("./"):
        audio_path = os.path.join(audio_dir, row[audio_key][2:])
    else:
        if row[audio_key].endswith(".wav"):
            audio_path = os.path.join(audio_dir, row[audio_key])
        else:
            audio_path = os.path.join(audio_dir, row[audio_key] + ".wav")
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ],
        },
    ]

def process_batch(model, processor, data: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Process a batch of data through the model"""
    results = []
    
    for row in tqdm(data):
        try:
            # Prepare conversation prompt
            conversation = create_prompt(row, args)
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=2048,
                stop=["</seg>"],  # 设置停止词
                skip_special_tokens=False
            )
            
            full_output = ""
            # Prepare inputs
            prompt = processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            audio_path = conversation[0]["content"][0]["audio"]
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            inputs = {
                "prompt": prompt[0],
                "multi_modal_data": {"audio": [(audio_data, sample_rate)]},
            }

            turn_time = 0
            while turn_time < 8:
                turn_time += 1
                outputs = model.generate(
                    prompts = inputs,
                    sampling_params = sampling_params
                )
                new_text = outputs[0].outputs[0].text

                # Check if the generation stopped at </seg>
                if outputs[0].outputs[0].stop_reason == "</seg>":
                    new_text += "</seg>"
                    full_output += new_text
                    timestamp_matches = list(re.finditer(r'<seg>([\d\.]+,\s*[\d\.]+)</seg>', new_text))

                    if timestamp_matches:
                        # Get the last timestamp match in this segment
                        last_match = timestamp_matches[-1]
                        timestamp = last_match.group(1)

                        start_time, end_time = map(float, timestamp.split(','))
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        
                        # Extract audio segment
                        segment = audio_data[start_sample:end_sample]
                        if len(segment) < 3200:
                            inputs["prompt"] += new_text
                            continue
                        
                        # Insert audio tokens after the last </seg> tag
                        processed_text = new_text[:last_match.end()] + "<|audio_bos|><|AUDIO|><|audio_eos|>" + new_text[last_match.end():]
                        
                        # Update the inputs with new audio segment
                        inputs["multi_modal_data"]["audio"].append((segment, sample_rate))
                        inputs["prompt"] += processed_text
                        
                        # Continue generation with new context
                    else:
                        inputs["prompt"] += new_text

                else:
                    full_output += new_text
                    break
            
            if row.get("choices"):
                prediction = extract_prediction(full_output, row["choices"])
            else:
                prediction = full_output.split("\nassistant\n")[-1]
            
            # Add prediction to result
            result = copy.deepcopy(row)
            result["model_response"] = full_output
            result["model_prediction"] = prediction
            results.append(result)
            
        except Exception as e:
            print(f"Error processing row {row.get('id', 'unknown')}: {str(e)}")
            # Add original row with error message if processing fails
            result = copy.deepcopy(row)
            result["model_response"] = f"ERROR: {str(e)}"
            result["model_prediction"] = ""
            results.append(result)
    
    return results



def analyse_response(text: str) -> str:
    """
    从字符串中提取 <think>...</think>和<answer>...</answer> 的内容，并返回第二个匹配项。
    如果没有找到足够的匹配项，返回空字符串或抛出异常（根据需求调整）。
    """
    # 使用正则表达式匹配所有 <think>...</think> 的内容
    think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)  # re.DOTALL 让 . 匹配换行符
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)  # re.DOTALL 让 . 匹配换行符
    
    think = ""
    answer = ""
    # 检查是否有至少两个匹配项
    if len(think_matches) >= 2:
        think = think_matches[1].strip()
    if len(answer_matches) >= 2:
        answer = answer_matches[1].strip()
    
    return think, answer


def extract_prediction(answer: str, choices: list) -> str:
    response = answer.split("\nassistant\n")[-1]
    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_matches:
        response = answer_matches[0].strip()

    for choice in choices:
        if choice.lower() in response.lower():
            return choice
    return choices[0]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5-Omni model")
    parser.add_argument("--checkpoint", type=str, default='/path/to/merged_model')
    parser.add_argument("--prompt", type=str, default='inference/grounding_1_0715.txt')
    parser.add_argument("--benchmark", type=str, default='MMAR')
    parser.add_argument("--output_file", type=str, default='output/test.json')
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--total_rank", type=int, default=8)
    args = parser.parse_args()

    if args.benchmark not in DATASET_REGISTER:
        raise ValueError(f"Benchmark {args.benchmark} not supported. Please choose from {list(DATASET_REGISTER.keys())}")
    
    # Load model
    print("Loading model...")
    model = LLM(model=args.checkpoint)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint)
    
    # Load data
    print("Loading data...")
    with open(DATASET_REGISTER[args.benchmark]["qa_file"], "r") as f:
        data = json.load(f)
    
    # Select subset of data
    if not isinstance(data, list):
        raise ValueError("Input JSON should be a list of items")
    
    start = args.rank * DATASET_REGISTER[args.benchmark]["num_samples"] // args.total_rank
    end = (args.rank + 1) * DATASET_REGISTER[args.benchmark]["num_samples"] // args.total_rank
    subset = data[start:end]
    print(f"Processing {len(subset)} items (indices {start}-{end-1})...")
    
    # Process data
    results = process_batch(model, processor, subset, args)
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Done!")