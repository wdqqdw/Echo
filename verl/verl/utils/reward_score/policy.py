import re
from typing import Dict, Optional

format_reward = 1

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """处理真实标签，提取判断结果和归因分析"""
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    lines = solution_text.split('\n')
    if lines:
        status_dict['judgment'] = lines[0].strip()
        status_dict['analysis'] = '\n'.join(lines[1:]).strip()
        print(f"  Found: judgment → {status_dict['judgment']}")
        print(f"  Found: analysis → {status_dict['analysis'][:50]}...")
    else:
        print("  [Warning] Unparseable ground truth")

    return status_dict

def validate_response_structure(processed_str: str) -> bool:
    """验证输出结构是否符合要求（跳过思考部分）"""
    print("\n[Structure Validation]")
    validation_passed = True

    # 新增：使用正则表达式提取正式回答部分（以'是/否'开头的内容）
    pattern = r'^([是否])\s*\n*(.*)'  # 匹配行首的是/否，允许空行分隔
    match = re.search(pattern, processed_str, re.MULTILINE)
    
    if not match:
        print("  [Error] No valid '是/否' judgment found")
        return False
    
    # 提取判断和分析部分
    judgment = match.group(1).strip()
    analysis = match.group(2).strip()
    
    if not judgment:
        print("  [Error] Missing '是/否' judgment")
        validation_passed = False
    if not analysis:
        print("  [Error] Missing attribution analysis")
        validation_passed = False
    
    if validation_passed:
        print("  Structure validation passed")
    else:
        print(f"  Extracted judgment: {judgment}, analysis: {analysis[:50]}...")
    
    return validation_passed

def parse_model_answer(answer_text: str) -> Optional[Dict[str, str]]:
    """解析模型输出（跳过思考内容，提取正式回答）"""
    status_dict = {}
    print("\n[Model Answer Parsing]")

    # 新增：使用正则表达式提取正式回答部分
    pattern = r'^([是否])\s*\n*(.*)'
    match = re.search(pattern, answer_text, re.MULTILINE)
    
    if not match:
        print("  [Error] Unparseable model answer (no '是/否' found)")
        return None
    
    status_dict['judgment'] = match.group(1).strip()
    status_dict['analysis'] = match.group(2).strip()
    
    print(f"  Extracted judgment → {status_dict['judgment']}")
    print(f"  Extracted analysis → {status_dict['analysis'][:50]}...")
    return status_dict

def compute_score(solution_str: str, ground_truth: str) -> float:
    """计算最终得分（跳过思考内容校验）"""
    gt_status = parse_solution_text_format(ground_truth)
    print(f"[Ground Truth] Final judgment: {gt_status['judgment']}")

    processed_str = solution_str.strip()
    print(f"\n[Model Response]\n{processed_str}")

    # 验证输出结构（新增跳过思考逻辑）
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    answer_score = 0
    if format_correct:
        pred_status = parse_model_answer(processed_str)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            if pred_status['judgment'] == gt_status['judgment']:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print("  Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors")

    total_score = format_score + answer_score
    return total_score