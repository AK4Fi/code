import pandas as pd
from collections import defaultdict, Counter
from math import log2


def main():
    # 读取数据
    df = pd.read_csv("totalstrain.csv")

    # ===== 第一阶段：计算全局信息熵 =====
    # 统计每个家族的opcode出现情况
    class_op_counts = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)

    # 遍历统计
    for _, row in df.iterrows():
        if not isinstance(row["text"], str):
            continue
        ops = row["text"].split()
        cls = row["Class"]
        for op in set(ops):  # 每个样本内相同opcode只计一次
            class_op_counts[cls][op] += 1
        total_class_counts[cls] += 1

    # 计算信息熵
    op_entropy = {}
    all_ops = set()
    for cls in class_op_counts.values():
        all_ops.update(cls.keys())

    for op in all_ops:
        probabilities = []
        for cls in total_class_counts.keys():
            count = class_op_counts[cls].get(op, 0)
            total = total_class_counts[cls]
            probabilities.append(count / total if total > 0 else 0)

        prob_sum = sum(probabilities)
        if prob_sum == 0:
            op_entropy[op] = 0.0
            continue
        norm_probs = [p / prob_sum for p in probabilities]
        entropy = -sum(p * log2(p) for p in norm_probs if p > 0)
        op_entropy[op] = entropy

    # 创建全局熵排名
    sorted_ops = sorted(op_entropy.items(), key=lambda x: x[1], reverse=True)
    op_rank = {op[0]: idx + 1 for idx, op in enumerate(sorted_ops)}

    # ===== 第二阶段：逐个样本分析 =====
    for idx, row in df.iterrows():
        print(f"\n=== 样本 {row['Id']} (类别 {row['Class']}) ===")

        if not isinstance(row["text"], str) or not row["text"].strip():
            print("无有效操作码序列")
            continue

        ops = row["text"].split()
        counter = Counter(ops)

        # 找出只出现一次的操作码及其位置
        single_ops = {
            op: [i for i, o in enumerate(ops) if o == op]
            for op, cnt in counter.items() if cnt == 1
        }

        if not single_ops:
            print("没有出现次数为1的操作码")
            continue

        # 生成报告
        report = []
        for op, positions in single_ops.items():
            entropy = op_entropy.get(op, 0)
            rank = op_rank.get(op, "N/A")
            report.append((
                op,
                positions[0],  # 因为只出现一次
                f"熵值 {entropy:.4f} (全局排名 {rank})"
            ))

        # 按熵值降序排序
        report.sort(key=lambda x: op_entropy.get(x[0], 0), reverse=True)

        # 格式化输出
        for op_info in report:
            print(f"操作码: {op_info[0]:<15} | 位置: {op_info[1]:<5} | {op_info[2]}")

        print(f"共找到 {len(report)} 个唯一出现操作码")


if __name__ == "__main__":
    main()