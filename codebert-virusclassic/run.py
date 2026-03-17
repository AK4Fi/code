# 全部流程代码
import numpy as np
import torch
from torchgen.api.types import longT
from transformers import BertTokenizer
import os
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('E:\pycharmcode\codebert-base')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [] #df['Class'].astype(int).tolist()  # 直接从DataFrame中提取数字标签列表，并转换为整数
        self.texts = []
        base_path = 'E:\\kaggledata\\subtraintxt\\'  # 确保路径分隔符正确
        for index, row in df.iterrows():
            filename = row['Id']
            label = row['Class'] - 1
            full_path = os.path.join(base_path, f"{filename}.txt")
            if not os.path.exists(full_path):
                print(f"Warning: File {full_path} does not exist. Skipping...")
                continue
            with open(full_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = text[:3500]
                # print(text)
                tokenized_text = tokenizer(text,
                                            padding='max_length',
                                            max_length=512,
                                            truncation=True,
                                            return_tensors="pt")
            self.texts.append(tokenized_text)
            self.labels.append(int(label))
            # print(self.texts)
        assert len(self.texts) == len(self.labels), "texts and labels must have the same length"

        # for filename in df['Id']:
        #     full_path = os.path.j9oin(base_path, f"{filename}.txt")  # 拼接文件路径
        #     if not os.path.exists(full_path):  # 检查文件是否存在
        #         print(f"Warning: File {full_path} does not exist. Skipping...")
        #         continue  # 如果文件不存在，跳过该文件
        #     with open(full_path, 'r', encoding='utf-8') as file:  # 打开文件
        #         text = file.read()  # 读取文件内容
        #         text = text[ :512]  # 截取前512个字符
        #         tokenized_text = tokenizer(text,
        #                                    padding='max_length',
        #                                    max_length=512,
        #                                    truncation=True,
        #                                    return_tensors="pt")
        #         self.texts.append(tokenized_text)


    # def __init__(self, df):
    #     self.labels = [labels[label] for label in df['category']]
    #     self.texts = [tokenizer(text,
    #                             padding='max_length',
    #                             max_length=512,
    #                             truncation=True,
    #                             return_tensors="pt")
    #                   for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


# 数据集准备
# 拆分训练集、验证集和测试集 8:1:1
import pandas as pd

subtrain_text_df = pd.read_csv('E:\kaggledata\subtrain\subtrainLabels.csv')
# print(subtrain_text_df.head())
df = pd.DataFrame(subtrain_text_df)
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
print(len(df_train), len(df_val), len(df_test))
# 720 90 90


# 构建模型
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('E:\pycharmcode\codebert-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer



from torch.optim import Adam
from tqdm import tqdm


def train(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

EPOCHS = 20
model = BertClassifier()
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)


# 测试模型
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


# 用测试数据集进行测试
evaluate(model, df_test)
torch.save(model.state_dict(),"berttextlong1.pth")