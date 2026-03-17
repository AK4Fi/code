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
            text = row['miniopcode']#miniopcode
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

subtrain_text_df = pd.read_csv('../data/strain_miniopcode.csv')#strain_miniopcode
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
        return final_layer, dropout_output
        #考虑到relu可能会影响训练和验证的效果，所以可以考虑删掉不删是93.8%，删掉是90.1%，不应该删
        # return linear_output, dropout_output


from torch.optim import Adam
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, train_data, val_data, learning_rate, epochs):
    # 定义中心损失类
    class CenterLoss(nn.Module):
        def __init__(self, num_classes=9, feat_dim=768, lambda_param=0.1):
            super().__init__()
            self.num_classes = num_classes
            self.feat_dim = feat_dim
            self.lambda_param = lambda_param
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))  # 可学习的类别中心

        def forward(self, features, labels):
            batch_size = features.size(0)
            # 提取对应类别的中心向量
            centers_batch = self.centers[labels]
            # 计算特征与中心的距离
            loss = self.lambda_param * 0.5 * torch.sum((features - centers_batch) ** 2) / batch_size
            return loss

    # 定义联合损失类
    class CombinedLoss(nn.Module):
        def __init__(self, num_classes=9, feat_dim=768, lambda_param=0.1):
            super().__init__()
            self.cross_entropy = nn.CrossEntropyLoss()
            self.center_loss = CenterLoss(num_classes, feat_dim, lambda_param)

        def forward(self, logits, features, labels):
            ce_loss = self.cross_entropy(logits, labels)
            center_loss = self.center_loss(features, labels)
            return ce_loss + center_loss

    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    # 设备判断
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 初始化联合损失（假设BERT输出的特征维度是768）
    criterion = CombinedLoss(num_classes=9, feat_dim=768, lambda_param=0.1)  # 核心修改点
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # 训练循环（需修改模型输出）
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            # 修改模型输出：需同时返回logits和特征
            # 假设模型返回格式为 (logits, features)
            output, features = model(input_id, mask)  # 需要修改模型forward函数

            # 计算联合损失
            batch_loss = criterion(output, features, train_label)  # 传入三个参数
            total_loss_train += batch_loss.item()

            # 精度计算保持不变
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # 验证阶段
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, _ = model(input_id, mask)  # 验证时只需logits
                batch_loss = criterion.cross_entropy(output, val_label)  # 验证阶段仅计算交叉熵损失
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
# 因为本案例中是处理多类分类问题，则使用分类交叉熵作为我们的损失函数。
EPOCHS = 20
model = BertClassifier()
LR = 1e-5
train(model, df_train, df_val, LR, EPOCHS)


# 测试模型
# def evaluate(model, test_data):
#     test = Dataset(test_data)
#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
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


def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()  # 确保模型处于评估模式

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            # 修改点：接收logits并忽略特征
            logits, _ = model(input_id, mask)  # 核心修改

            # 计算精度（使用logits）
            acc = (logits.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# 用测试数据集进行测试
evaluate(model, df_test)
# torch.save(model.state_dict(),"berttextlong1.pth")
