# 全部流程代码
import numpy as np
import torch
from torchgen.api.types import longT
from transformers import BertTokenizer
import os
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = []
        self.texts = []
        for index, row in df.iterrows():
            label = row['Class'] - 1
            text = row['opcodes']#miniopcode
            tokenized_text = tokenizer(text,
                                        padding='max_length',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt")
            self.texts.append(tokenized_text)
            self.labels.append(int(label))

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


import pandas as pd

subtrain_text_df = pd.read_csv('../data/totalopcode_re.csv')#strain_miniopcode
df = pd.DataFrame(subtrain_text_df)
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
print(len(df_train), len(df_val), len(df_test))

from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('../model/codebert-base')
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

# def train(model, train_data, val_data, learning_rate, epochs):
#     # 通过Dataset类获取训练和验证集
#     train, val = Dataset(train_data), Dataset(val_data)
#     # DataLoader根据batch_size获取数据，训练时选择打乱样本
#     train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
#     val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)
#     # 判断是否使用GPU
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#
#     if use_cuda:
#         model = model.cuda()
#         criterion = criterion.cuda()
#         print("ues cuda")
#     # 开始进入训练循环
#     for epoch_num in range(epochs):
#         # 定义两个变量，用于存储训练集的准确率和损失
#         total_acc_train = 0
#         total_loss_train = 0
#         # 进度条函数tqdm
#         for train_input, train_label in tqdm(train_dataloader):
#             train_label = train_label.to(device)
#             mask = train_input['attention_mask'].to(device)
#             input_id = train_input['input_ids'].squeeze(1).to(device)
#             # 通过模型得到输出
#             output = model(input_id, mask)
#             # 计算损失
#             batch_loss = criterion(output, train_label.long())
#             total_loss_train += batch_loss.item()
#             # 计算精度
#             acc = (output.argmax(dim=1) == train_label).sum().item()
#             total_acc_train += acc
#             # 模型更新
#             model.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#         # ------ 验证模型 -----------
#         # 定义两个变量，用于存储验证集的准确率和损失
#         total_acc_val = 0
#         total_loss_val = 0
#         # 不需要计算梯度
#         with torch.no_grad():
#             # 循环获取数据集，并用训练好的模型进行验证
#             for val_input, val_label in val_dataloader:
#                 # 如果有GPU，则使用GPU，接下来的操作同训练
#                 val_label = val_label.to(device)
#                 mask = val_input['attention_mask'].to(device)
#                 input_id = val_input['input_ids'].squeeze(1).to(device)
#
#                 output = model(input_id, mask)
#
#                 batch_loss = criterion(output, val_label.long())
#                 total_loss_val += batch_loss.item()
#
#                 acc = (output.argmax(dim=1) == val_label).sum().item()
#                 total_acc_val += acc
#
#         print(
#             f'''Epochs: {epoch_num + 1}
#               | Train Loss: {total_loss_train / len(train_data): .3f}
#               | Train Accuracy: {total_acc_train / len(train_data): .3f}
#               | Val Loss: {total_loss_val / len(val_data): .3f}
#               | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
# 首先在文件开头导入所需库
from sklearn.metrics import f1_score, classification_report


# 修改训练函数
def train(model, train_data, val_data, learning_rate, epochs):
    train_dataset, val_dataset = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        # 训练阶段
        model.train()
        total_loss_train, total_acc_train = 0, 0
        all_preds, all_labels = [], []  # 新增：收集预测和标签

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            output = model(input_id, mask)

            # 收集预测结果
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())  # 注意设备转移
            all_labels.extend(train_label.cpu().numpy())

            # 原有损失计算
            loss = criterion(output, train_label)
            total_loss_train += loss.item()
            acc = (preds == train_label).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()

        # 计算训练集F1
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        train_micro_f1 = f1_score(all_labels, all_preds, average='micro')

        # 验证阶段
        model.eval()
        total_loss_val, total_acc_val = 0, 0
        val_preds, val_labels = [], []  # 新增：验证集预测收集

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                preds = torch.argmax(output, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(val_label.cpu().numpy())

                loss = criterion(output, val_label)
                total_loss_val += loss.item()
                acc = (preds == val_label).sum().item()
                total_acc_val += acc

        # 计算验证集F1
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_micro_f1 = f1_score(val_labels, val_preds, average='micro')

        # 打印结果（添加F1显示）
        print(f'''
        Epoch {epoch_num + 1}
        Train Loss: {total_loss_train / len(train_data):.3f} | Acc: {total_acc_train / len(train_data):.3f}
        Train Macro-F1: {train_macro_f1:.3f} | Micro-F1: {train_micro_f1:.3f}
        Val Loss: {total_loss_val / len(val_data):.3f} | Acc: {total_acc_val / len(val_data):.3f}
        Val Macro-F1: {val_macro_f1:.3f} | Micro-F1: {val_micro_f1:.3f}
        ''')


# 修改测试函数
def evaluate(model, test_data):
    test_dataset = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    total_acc_test = 0
    all_preds, all_labels = [], []  # 新增

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_label.cpu().numpy())

            acc = (preds == test_label).sum().item()
            total_acc_test += acc

    # 计算详细分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i + 1}" for i in range(9)]))

    # 计算综合F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    print(f'''Test Results:
    Accuracy: {total_acc_test / len(test_data):.3f}
    Macro-F1: {macro_f1:.3f}
    Micro-F1: {micro_f1:.3f}''')

EPOCHS = 10
model = BertClassifier()
LR = 1e-5
train(model, df_train, df_val, LR, EPOCHS)


# 测试模型
# def evaluate(model, test_data):
#     test = Dataset(test_data)
#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     if use_cuda:
#         model = model.cuda()
#
#     total_acc_test = 0
#     with torch.no_grad():
#         for test_input, test_label in test_dataloader:
#             test_label = test_label.to(device)
#             mask = test_input['attention_mask'].to(device)
#             input_id = test_input['input_ids'].squeeze(1).to(device)
#             output = model(input_id, mask)
#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
#     print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# 用测试数据集进行测试
evaluate(model, df_test)
# torch.save(model.state_dict(),"berttextlong1.pth")