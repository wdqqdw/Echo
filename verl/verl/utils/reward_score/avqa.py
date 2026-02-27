# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch

def compute_score(solution_str, ground_truth, id, data_item, tokenizer):
    """The scoring function for AVQA.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    seg_count, total_duration, non_overlapping_time, avg_duration, inconsistent_count = extract_time_info(solution_str)

    think_matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)  # re.DOTALL 让 . 匹配换行符
    answer_matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)  # re.DOTALL 让 . 匹配换行符
    if len(think_matches) != 1 or len(answer_matches) != 1:
        format_score = 0
        acc_score = 0
    else:
        format_score = 0.5
        answer = answer_matches[0].strip()

        if data_item.batch.get('old_log_probs') is None:
            #Validation
            if answer == ground_truth:
                acc_score = 0.5
            else:
                acc_score = 0
        else:
            #Training
            target_index = find_answer_pattern_flexible(data_item.batch['responses'], tokenizer, answer, ground_truth)
            confidence = torch.exp(data_item.batch['old_log_probs'][target_index]).item()
            
            if id in ["multiturn_rl_6", "multiturn_rl_11", "multiturn_rl_12"]:
                if answer == ground_truth:
                    acc_score = 0.5*(1-(confidence-1)**2)
                else:
                    acc_score = -0.5*(confidence**2)
            elif id == "multiturn_rl_10":
                if answer == ground_truth:
                    acc_score = 0.5*(confidence)
                else:
                    acc_score = -0.5*(confidence)
            else:
                if answer == ground_truth:
                    acc_score = 0.5
                else:
                    acc_score = 0

    tool_score = 0
    if acc_score > 0:
        if id == "multiturn_rl_11":
            valid_seg_count = min(seg_count, 2)
            tool_score = acc_score * valid_seg_count / 2
        elif id == "multiturn_rl_12":
            valid_seg_count = min(seg_count, 3)
            tool_score = acc_score * valid_seg_count / 3
        elif seg_count > 0:
            tool_score = acc_score

    inconsistent_count = min(inconsistent_count, 5)
    consistency_score = 0 - inconsistent_count / 10

    if total_duration > 0:
        non_overlap_score = 0.5 * non_overlapping_time / total_duration
    else:
        non_overlap_score = 0


    # import pdb
    # pdb.set_trace()
    # tokenizer.encode("<answer>") [27, 9217, 29]
    # tokenizer.encode("</answer>") [522, 9217, 29]
    # tokenizer.encode("<think>") [13708, 766, 29]
    # tokenizer.encode("</think>") [522, 26865, 29]

    # selected_response = data_item.batch['responses'][data_item.batch['response_mask'].bool()].size()
    # data_item.batch['old_log_probs']
    # data_item.batch['rollout_log_probs'][:300]
    # torch.exp(data_item.batch['old_log_probs'][:300])

    if id == "multiturn_rl_1":
        total_reward = format_score + acc_score + consistency_score
    elif id == "multiturn_rl_2":
        total_reward = format_score + acc_score + tool_score
    elif id in ["multiturn_rl_3", "multiturn_rl_6"]:
        total_reward = format_score + acc_score + tool_score + consistency_score
    elif id in ["multiturn_rl_11", "multiturn_rl_12"]:
        total_reward = format_score + acc_score + tool_score + consistency_score + non_overlap_score
    else:
        total_reward = format_score + acc_score
    return {
        "score": total_reward,
        "acc_score": acc_score,
        "format_score": format_score,
        "seg_count": seg_count,
        "total_duration": total_duration,
        "non_overlapping_time": non_overlapping_time,
        "avg_duration": avg_duration,
        "tool_score": tool_score,
        "consistency_score": consistency_score,
        "non_overlap_score": non_overlap_score
    }


def extract_time_info(text: str):
    """
    从文本中提取时间信息并计算统计量
    
    参数:
        text: 包含时间段的文本
        
    返回:
        元组包含:
        - 总匹配个数
        - 总时间(秒)
        - 不重叠时间(秒)
        - 平均时间长度(秒)
        - 后接大写字母或'<'的匹配个数
    """
    # 正则表达式匹配<seg>标签中的时间对及后面的一个字符
    pattern = r'<seg>([\d\.]+,\s*[\d\.]+)</seg>(.?)[^<]?'
    matches = re.findall(pattern, text)
    
    if not matches:
        return 0, 0.0, 0.0, 0.0, 0
    
    # 解析所有时间区间并检查后续字符
    intervals = []
    special_case_count = 0
    
    for match in matches:
        time_pair, next_char = match
        start, end = map(float, time_pair.replace(' ', '').split(','))
        intervals.append((start, end))
        
        # 检查后续字符是否是大写字母或'<'
        if next_char and (next_char.isupper() or next_char == '<'):
            special_case_count += 1
    
    # 计算总匹配个数
    total_count = len(intervals)
    
    # 计算总时间和平均时间
    total_duration = sum(end - start for start, end in intervals)
    avg_duration = total_duration / total_count if total_count > 0 else 0.0
    
    # 计算不重叠时间
    non_overlapping_time = 0.0
    if intervals:
        # 按开始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        
        # 合并重叠区间
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 有重叠
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        # 计算合并后的总时间
        non_overlapping_time = sum(end - start for start, end in merged)
    
    return total_count, total_duration, non_overlapping_time, avg_duration, special_case_count



def find_answer_pattern_flexible(tensor, tokenizer, solution_str, ground_truth):
    """
    灵活版本的字符串模式查找，支持一个ID对应多个字符的情况
    
    Args:
        tensor: 一维tensor，包含编码的数据
        decode_func: 解码函数，将tensor元素转换为字符串（可能一个ID对应多个字符）
        solution_str: 已有的解决方案字符串
        ground_truth: 目标真实字符串
    
    Returns:
        int: 匹配的索引位置，如果没找到返回-1
    """
    target_pattern = "<answer>"
    
    for i in range(len(tensor)):
        current_text = ""
        
        # 构建当前开始的字符串
        for j in range(i, len(tensor)):
            # 解码当前元素（可能得到多个字符）
            decoded_chars = tokenizer.decode(tensor[j])
            current_text += decoded_chars
            
            # 关键修改：检查current_text的任何后缀是否匹配target_pattern的起始
            # 而不仅仅是current_text本身


            has_valid_prefix = False
            if target_pattern in current_text:
                has_valid_prefix = True
            else:
                for k in range(len(current_text)):
                    suffix = current_text[k:]
                    if target_pattern.startswith(suffix):
                        has_valid_prefix = True
                        break
            
            if not has_valid_prefix:
                break
            
            # 检查是否完全包含了target_pattern（作为子字符串，不一定是精确前缀）
            if target_pattern in current_text:
                # import pdb
                # pdb.set_trace()
                # 找到target_pattern在current_text中的位置
                pattern_pos = current_text.find(target_pattern)
                content_after_pattern = current_text[pattern_pos + len(target_pattern):]
                if len(content_after_pattern) == 0:
                    continue
                
                # 检查solution和ground_truth
                if solution_str == ground_truth:
                    return j
                
                # 检查content_after_pattern是否匹配ground_truth的起始
                if ground_truth.startswith(content_after_pattern):

                    # 继续读取直到出现不匹配
                    remaining_content = content_after_pattern
                    k = j + 1
                    
                    while k < len(tensor) and ground_truth.startswith(remaining_content):
                        if remaining_content == ground_truth:
                            # import pdb
                            # pdb.set_trace()
                            return k

                        next_chars = tokenizer.decode(tensor[k])
                        remaining_content += next_chars
                        k += 1
                        
                        # 如果不再匹配ground_truth的起始部分
                        if not ground_truth.startswith(remaining_content):
                            # import pdb
                            # pdb.set_trace()
                            return k-1
                                
                    raise ValueError("Internal error: should not reach here")
                else:
                    # 如果不匹配ground_truth的起始部分
                    return j
                
        
    return -1