import re
from typing import Tuple

def extract_time_info(text: str) -> Tuple[int, float, float, float, int]:
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

# 示例使用
if __name__ == "__main__":
    sample_text = """
    Here is some text <seg>12.3, 14.5</seg> with time segments.
    Another segment <seg>15.0,17.2</seg>X and overlapping <seg>16.8,18.9</seg>.
    Non-overlapping <seg>20.0,22.0</seg><123 and single <seg>25.5,25.8</seg>A.
    <seg>30.0,32.0</seg>Bla bla <seg>35.0,37.0</seg>Z.
    """
    
    count, total, non_overlap, avg, special = extract_time_info(sample_text)
    
    print(f"总匹配个数: {count}")
    print(f"总时间: {total:.2f}秒")
    print(f"不重叠时间: {non_overlap:.2f}秒")
    print(f"平均时间长度: {avg:.2f}秒")
    print(f"后接大写字母或'<'的匹配个数: {special}")