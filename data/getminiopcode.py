import pandas as pd
from collections import Counter


def process_mini_opcodes(text):
    """循环处理操作码直到长度≤600的核心函数"""
    if pd.isna(text) or not str(text).strip():
        return ""

    opcodes = str(text).strip().split()

    # 循环处理直到满足长度要求
    while len(opcodes) > 600:
        # 统计当前操作码频率
        counter = Counter(opcodes)

        # 获取当前最高频的1个操作码
        if not counter:
            break  # 防止空计数器
        top3_ops = [item[0] for item in counter.most_common(1)]

        # 过滤掉这三个高频操作码的所有出现
        opcodes = [op for op in opcodes if op not in top3_ops]

        # 安全保护：如果所有操作码都被删除则终止循环
        if len(opcodes) == 0:
            break

    return " ".join(opcodes)


# 读取原始数据
df = pd.read_csv("kagglesample.csv")

# 应用处理函数
df["miniopcodes"] = df["opcodes"].apply(process_mini_opcodes)

# 保存结果文件
df[["Id", "Class", "miniopcodes"]].to_csv("kagglemini600.csv", index=False)