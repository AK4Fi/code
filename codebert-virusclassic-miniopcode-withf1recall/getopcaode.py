import re
import os


def getOpcodeSequence(filename):
    """提取操作码序列并过滤无效指令"""
    opcode_seq = []
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    exclude_ops = {"align", "db", "byte"}

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
input_dir = r"E:\kaggledata\subtrain"
output_dir = r"E:\kaggledata\subtraintxt"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历所有.asm文件
for filename in os.listdir(input_dir):
    if not filename.endswith(".asm"):
        continue

    # 处理单个文件
    input_path = os.path.join(input_dir, filename)
    opcodes = getOpcodeSequence(input_path)

    # 跳过空结果
    if not opcodes:
        continue

    # 生成输出路径
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    # 写入操作码到TXT文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(opcodes))