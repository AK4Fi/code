import pandas as pd
from collections import Counter

# 读取合并后的文件
df = pd.read_csv('totalopcode2015_re.csv')


# 定义操作码统计函数
def count_opcodes(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""

    # 默认按空格分割操作码
    opcodes = str(text).strip().split()
    counter = Counter(opcodes)

    # 将统计结果格式化为字符串（示例格式："MOV:3,JMP:2,ADD:1"）
    return ",".join([f"{k}:{v}" for k, v in counter.items()])


# 应用统计函数并创建rate列
df['rate'] = df['opcodes'].apply(count_opcodes)

# 提取目标列并保存
df[['Id', 'Class', 'rate']].to_csv('totalopcode2015_re_rate.csv', index=False)