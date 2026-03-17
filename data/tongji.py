import csv
import sys
from collections import defaultdict


def analyze_large_csv(csv_path):
    # 调整CSV字段大小限制（解决大字段问题）
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    # 初始化统计容器
    class_counts = defaultdict(int)
    total_length = 0
    max_length = 0
    total_samples = 0
    opcodes = set()

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 使用迭代器方式读取（内存友好）
            reader = csv.reader(f)

            try:
                headers = next(reader)  # 跳过标题行
            except StopIteration:
                print("CSV文件为空")
                return

            for row_num, row in enumerate(reader, 2):  # 行号从2开始计数
                try:
                    # 验证行数据完整性
                    if len(row) < 3:
                        print(f"第 {row_num} 行数据不完整，已跳过")
                        continue

                    # 统计类别
                    class_value = row[1].strip()
                    class_counts[class_value] += 1

                    # 处理操作码序列
                    opcode_str = row[2].strip()
                    if not opcode_str:
                        continue

                    # 流式处理操作码（避免内存爆炸）
                    seq_length = 0
                    current_opcodes = []
                    for op in opcode_str.split():
                        seq_length += 1
                        opcodes.add(op)
                        current_opcodes.append(op)

                    # 更新统计信息
                    total_length += seq_length
                    total_samples += 1
                    if seq_length > max_length:
                        max_length = seq_length

                    # 及时清理临时数据
                    del current_opcodes

                except Exception as e:
                    print(f"处理第 {row_num} 行时发生错误：{str(e)}")
                    continue

    except UnicodeDecodeError:
        print("文件编码错误，请尝试使用其他编码（如latin-1）")
        return

    # 计算结果
    average_length = total_length / total_samples if total_samples > 0 else 0
    sorted_opcodes = sorted(opcodes)

    # 输出统计结果
    print("\n=== 类别分布统计 ===")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls.ljust(15)}: {count:>6} 个样本")

    print("\n=== 操作码序列统计 ===")
    print(f"总样本数     : {total_samples:>8}")
    print(f"平均长度     : {average_length:>8.1f} 操作码")
    print(f"最大长度     : {max_length:>8} 操作码")

    print("\n=== 操作码统计 ===")
    print(f"唯一操作码数量: {len(sorted_opcodes):>8}")
    print("示例操作码（total）:")
    print(', '.join(sorted_opcodes[:1000]))


# 使用示例（文件路径需要修改为实际路径）
csv_path = r"kagglemini600.csv"
analyze_large_csv(csv_path)