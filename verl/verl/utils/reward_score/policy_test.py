import torch
import pandas as pd

# 定义计算 F1 分数的函数
def compute_score(solution_str: str, ground_truth: str) -> float:
    # 解析真实标签
    gt_lines = ground_truth.split('\n')
    if not gt_lines or gt_lines[0].strip() not in ['是', '否']:
        return 0
    gt_judgment = gt_lines[0].strip()

    # 解析模型输出
    pred_lines = solution_str.split('\n')
    if not pred_lines or pred_lines[0].strip() not in ['是', '否']:
        return 0
    pred_judgment = pred_lines[0].strip()

    # 计算 TN, TP, FN, FP
    if gt_judgment == '是' and pred_judgment == '是':
        tp = 1
        fn = 0
        tn = 0
        fp = 0
    elif gt_judgment == '是' and pred_judgment == '否':
        tp = 0
        fn = 1
        tn = 0
        fp = 0
    elif gt_judgment == '否' and pred_judgment == '是':
        tp = 0
        fn = 0
        tn = 0
        fp = 1
    else:
        tp = 0
        fn = 0
        tn = 1
        fp = 0

    # 计算 F1 分数
    if tp + fp == 0 or tp + fn == 0:
        f1_score = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score
# 