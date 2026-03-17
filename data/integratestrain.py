import pandas as pd

# 定义列名（与原始文件保持一致）
columns = ['Id', 'Class', 'text']

# 读取第一个文件（保留标题行）
train_df = pd.read_csv('train.csv')

# 读取其他两个文件（跳过标题行）
del_df = pd.read_csv('dev.csv', header=None, skiprows=1, names=columns)
text_df = pd.read_csv('test.csv', header=None, skiprows=1, names=columns)

# 合并三个DataFrame
combined_df = pd.concat([train_df, del_df, text_df], ignore_index=True)

# 保存合并后的文件
combined_df.to_csv('totalstrain.csv', index=False)