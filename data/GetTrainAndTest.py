import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置随机种子
random.seed(112)
np.random.seed(112)

try:
    # 读取CSV文件
    df = pd.read_csv('totalopcode2015-rekeytext3500.csv')
    df = df.dropna()

    # 首先将数据集分为训练集和剩余集，比例为0.8:0.2
    train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=112)

    # 然后将剩余集分为开发集和测试集，比例为0.5:0.5
    dev_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=112)

    # 保存训练集、开发集和测试集
    train_df.to_csv('totalopcode2015-rekeytext3500_train.csv', index=False)
    dev_df.to_csv('totalopcode2015-rekeytext3500_dev.csv', index=False)
    test_df.to_csv('totalopcode2015-rekeytext3500_test.csv', index=False)

    print("数据分割完成，训练集、开发集和测试集已保存。")

except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")
