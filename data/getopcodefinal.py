import re
import os
import csv
import time
time_start = time.time() #开始计时
def getOpcodeSequence(filename):
    """提取操作码序列并过滤无效指令"""
    opcode_seq = []
    # 修改正则表达式：只匹配2个及以上字母的操作码 ([a-z]{2,})
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]{2,})')  # <-- 关键修改
    exclude_ops = {"align", "db", "byte","dd","word", "dword"}

    with open(filename, mode="r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith(".text"):
                m = re.findall(p, line)
                if m:
                    opc = m[0][1]
                    if opc not in exclude_ops:
                        opcode_seq.append(opc)
    return opcode_seq

# 输入输出路径配置
input_dir = r"E:\kaggledata\subtrain"      # 包含原始CSV和.asm文件的目录
output_dir = r"E:\pycharmcode\data"  # 新CSV文件输出目录
input_csv_name = "subtrainLabels.csv"               # 原始CSV文件名（位于input_dir中）
output_csv_name = "kagglesample.csv"     # 新CSV文件名

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 输入输出CSV路径
input_csv_path = os.path.join(input_dir, input_csv_name)
output_csv_path = os.path.join(output_dir, output_csv_name)

# 处理CSV文件
with open(input_csv_path, 'r', encoding='utf-8', newline='') as f_in, \
     open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out:

    csv_reader = csv.reader(f_in)
    csv_writer = csv.writer(f_out)

    # 写入新CSV的标题行（假设原始CSV有标题）
    headers = next(csv_reader)  # 读取原始CSV的标题
    csv_writer.writerow(headers + ["opcodes"])  # 添加新列标题

    for row in csv_reader:
        if len(row) < 2:  # 跳过不完整行
            continue

        filename = row[0].strip()  # 原始文件名（无后缀）
        file_class = row[1].strip()
        asm_file_path = os.path.join(input_dir, f"{filename}.asm")  # 构造.asm文件路径

        # 检查文件是否存在且非空
        if not os.path.exists(asm_file_path):
            continue

        # 提取操作码序列
        opcodes = getOpcodeSequence(asm_file_path)
        if not opcodes:
            continue

        # 写入新CSV
        csv_writer.writerow([filename, file_class, ' '.join(opcodes)])

time_end = time.time()    #结束计时
time_c= time_end - time_start   #运行所花时间
print('time cost', time_c, 's')