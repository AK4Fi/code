import pandas as pd
from collections import defaultdict, Counter
from math import log, log2


def renyi_entropy(probabilities, alpha=1.0):
    """
    计算Rényi熵
    :param probabilities: 概率列表
    :param alpha: 熵阶数参数
    :return: Rényi熵值
    """
    if alpha == 1:  # 退化为香农熵
        return -sum(p * log2(p) for p in probabilities if p > 0)
    elif alpha == 0:  # Hartley熵（最大熵）
        return log2(len([p for p in probabilities if p > 0])) if any(p > 0 for p in probabilities) else 0.0
    elif alpha == float('inf'):  # 最小熵
        return -log2(max(probabilities))
    else:
        sum_p_alpha = sum(p ** alpha for p in probabilities if p > 0)
        if sum_p_alpha <= 0:
            return 0.0
        return (1 / (1 - alpha)) * log2(sum_p_alpha)


def main(alpha=1.0):
    # 读取数据
    df = pd.read_csv("totalstrain.csv")

    # ===== 第一阶段：计算全局Rényi熵 =====
    class_op_counts = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)

    # 遍历统计
    for _, row in df.iterrows():
        if not isinstance(row["text"], str):
            continue
        ops = row["text"].split()
        cls = row["Class"]
        for op in set(ops):  # 使用set避免重复计数
            class_op_counts[cls][op] += 1
        total_class_counts[cls] += 1

    # 计算Rényi熵
    op_entropy = {}
    all_ops = set()
    for cls in class_op_counts.values():
        all_ops.update(cls.keys())

    print(f"\n总共有 {len(all_ops)} 个不同的操作码 (使用α={alpha}的Rényi熵)\n")

    for op in all_ops:
        probabilities = []
        for cls in total_class_counts.keys():
            count = class_op_counts[cls].get(op, 0)
            total = total_class_counts[cls]
            probabilities.append(count / total if total > 0 else 0)

        # 过滤零概率（避免数学错误）
        valid_probs = [p for p in probabilities if p > 0]
        entropy = renyi_entropy(valid_probs, alpha) if valid_probs else 0.0
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

        # 生成报告并添加样本内排名
        report = []
        for op, positions in single_ops.items():
            entropy = op_entropy.get(op, 0)
            global_rank = op_rank.get(op, "N/A")
            report.append((
                op,
                positions[0],  # 仅出现一次的位置
                entropy,
                global_rank
            ))

        # 按熵值降序排序（样本内排序）
        report.sort(key=lambda x: x[2], reverse=True)

        # 格式化输出
        for rank, (op, pos, entropy, g_rank) in enumerate(report, start=1):
            print(f"操作码: {op:<15} | 位置: {pos:<5} | "
                  f"Rényi熵 (α={alpha}) {entropy:.4f} | "
                  f"全局排名 {g_rank} | 样本排名 {rank}")

        print(f"\n共找到 {len(report)} 个唯一出现操作码")


if __name__ == "__main__":
    # 通过修改alpha参数控制熵类型
    # alpha=1: 香农熵, alpha=0: Hartley熵, alpha=2: 碰撞熵等
    #α=0：Hartley熵（最大熵）α=1：香农熵α=2：碰撞熵α=∞：最小熵
    main(alpha=1)  # 示例使用α=2的Rényi熵