import csv
import sys
import os


def truncate_opcodes(input_csv, output_csv):
    # 调整CSV字段大小限制
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None

    with open(input_csv, 'r', encoding='utf-8', errors='replace') as f_in, \
            open(output_csv, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # 处理标题行
        try:
            headers = next(reader)
            if len(headers) < 3:
                raise ValueError("输入文件列数不足")
            headers[2] = "opcodes512"  # 修改第三列标题
            writer.writerow(headers)
        except StopIteration:
            print("输入文件为空")
            return

        # 统计处理结果
        processed_count = 0
        skipped_count = 0

        for row in reader:
            try:
                # 验证行数据完整性
                if len(row) < 3:
                    skipped_count += 1
                    continue

                filename = row[0]
                file_class = row[1]
                opcodes_str = row[2].strip()

                # 处理操作码
                opcodes_list = opcodes_str.split()
                truncated = opcodes_list[:3500]  # 截取前512个
                new_opcodes = ' '.join(truncated)

                # 跳过空结果
                if not new_opcodes:
                    skipped_count += 1
                    continue

                # 写入新行
                writer.writerow([filename, file_class, new_opcodes])
                processed_count += 1

            except Exception as e:
                print(f"处理行时发生错误：{str(e)}")
                skipped_count += 1
                continue

        # 输出统计信息
        print(f"处理完成！共处理 {processed_count} 条有效记录")
        print(f"跳过 {skipped_count} 条无效记录")


# 使用示例
if __name__ == "__main__":
    input_path = r"totalopcode2015-rekeytext.csv"
    output_path = r"totaltotalopcode2015-rekeytext3500.csv"

    truncate_opcodes(input_path, output_path)