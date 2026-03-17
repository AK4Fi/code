# 全部流程代码
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torchgen.api.types import longT
from transformers import BertTokenizer
import os
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from torch import nn
from collections import defaultdict
import math
tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')
def calculate_renyi_weights(df, alpha=2, top_k=10):
    """计算全局Rényi熵并返回top-k高熵操作码的权重字典"""
    class_op_counts = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)

    for _, row in df.iterrows():
        if not isinstance(row["miniopcode"], str):
            continue
        ops = row["miniopcode"].split()
        cls = row["Class"] - 1
        for op in set(ops):
            class_op_counts[cls][op] += 1
        total_class_counts[cls] += 1
    op_entropy = {}
    all_ops = set()
    for cls in class_op_counts.values():
        all_ops.update(cls.keys())
    for op in all_ops:
        probabilities = []
        for cls in total_class_counts.keys():
            count = class_op_counts[cls].get(op, 0)
            total = total_class_counts[cls]
            probabilities.append(count / total if total > 0 else 0)
        valid_probs = [p for p in probabilities if p > 0]
        if not valid_probs:
            op_entropy[op] = 0.0
            continue
        if alpha == 1:
            entropy = -sum(p * math.log(p, 2) for p in valid_probs)
        else:
            sum_p_alpha = sum(p ** alpha for p in valid_probs)
            entropy = (1 / (1 - alpha)) * math.log(sum_p_alpha, 2)
        op_entropy[op] = entropy
    sorted_ops = sorted(op_entropy.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {op: 3 for op, _ in sorted_ops}  # 权重增强系数设为3倍
class RenyiBertClassifier(nn.Module):
    def __init__(self, renyi_weights, dropout=0.5):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('../model/codebert-base', attn_implementation="eager")
        self.tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.relu = nn.ReLU()
        self.renyi_weights = renyi_weights
        self.enhance_factor = 1.5  # 注意力增强系数
        self.op_to_tokens = {
            op: self.tokenizer.encode(op, add_special_tokens=False)
            for op in renyi_weights.keys()
        }
    def create_enhance_mask(self, input_ids):
        batch_size, seq_len = input_ids.shape
        mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        all_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.cpu().numpy()]
        for b in range(batch_size):
            tokens = all_tokens[b]
            for pos, token in enumerate(tokens):
                if token in self.tokenizer.special_tokens_map.values():
                    continue
                for op, op_tokens in self.op_to_tokens.items():
                    if pos >= len(op_tokens):
                        continue
                    if tokens[pos:pos + len(op_tokens)] == op_tokens:
                        mask[b, pos:pos + len(op_tokens)] = self.enhance_factor
                        break
        return mask.unsqueeze(1)  # [batch, 1, seq_len]
    def forward(self, input_ids, attention_mask, return_attention=False):
        enhance_mask = self.create_enhance_mask(input_ids)
        enhance_mask = enhance_mask.unsqueeze(2).repeat(1, 1, attention_mask.size(-1), 1)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
            output_hidden_states = True,
        )
        last_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden_size]
        enhanced_attention = last_attention * enhance_mask
        attention_probs = nn.functional.softmax(enhanced_attention, dim=-1)
        context = torch.matmul(attention_probs, hidden_states.unsqueeze(1))  # [batch, heads, seq, hidden]
        pooled_output = context.mean(dim=2)  # [batch, heads, hidden]
        pooled_output = pooled_output.mean(dim=1)  # [batch, hidden]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        if return_attention:
            return self.relu(linear_output), attention_probs
        else:
            return self.relu(linear_output) # 输出形状 [batch, num_classes]


subtrain_text_df = pd.read_csv('../data/strain_miniopcode.csv')
renyi_weights = calculate_renyi_weights(subtrain_text_df, alpha=2)
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
        return final_layer
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_curves(history):
    """绘制训练指标曲线"""
    plt.figure(figsize=(15, 10))

    # 动态获取实际完成的epoch数
    num_epochs = len(history['train_acc'])
    x = range(1, num_epochs + 1)

    # 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(x, history['train_acc'], 'b-', label='Train')
    plt.plot(x, history['val_acc'], 'r-', label='Validation')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # F1值曲线
    plt.subplot(2, 2, 2)
    plt.plot(x, history['train_f1_macro'], 'b--', label='Train Macro')
    plt.plot(x, history['val_f1_macro'], 'r--', label='Validation Macro')
    plt.plot(x, history['train_f1_weighted'], 'b:', label='Train Weighted')
    plt.plot(x, history['val_f1_weighted'], 'r:', label='Validation Weighted')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()

    # Recall曲线（宏平均）
    plt.subplot(2, 2, 3)
    plt.plot(x, history['train_recall_macro'], 'b-', label='Train')
    plt.plot(x, history['val_recall_macro'], 'r-', label='Validation')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Recall (Macro)')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Recall曲线（加权平均）
    plt.subplot(2, 2, 4)
    plt.plot(x, history['train_recall_weighted'], 'b-', label='Train')
    plt.plot(x, history['val_recall_weighted'], 'r-', label='Validation')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Recall (Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.close()
def train_enhanced(model, train_data, val_data, learning_rate, epochs):
    # 初始化历史记录
    history = {
        'train_acc': [],
        'train_f1_macro': [],
        'train_f1_weighted': [],
        'train_recall_macro': [],
        'train_recall_weighted': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'val_recall_macro': [],
        'val_recall_weighted': []
    }
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        print("torch.cuda.is_available() =", torch.cuda.is_available())
        print("ues cuda")
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        all_preds_train = []
        all_labels_train = []
        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            labels = labels.to(device)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1)
            all_preds_train.extend(preds.cpu().numpy())  # 保存预测结果
            all_labels_train.extend(labels.cpu().numpy())  # 保存真实标签
            assert outputs.dim() == 2, f"输出维度异常: {outputs.shape}"
            assert outputs.size(1) == 9, f"类别数不符: {outputs.size(1)}"
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).sum().item()
        train_f1_macro = f1_score(all_labels_train, all_preds_train, average='macro',zero_division=0)  # 宏平均
        train_f1_weighted = f1_score(all_labels_train, all_preds_train, average='weighted',zero_division=0)
        train_recall_macro = recall_score(all_labels_train, all_preds_train, average='macro',zero_division=0)
        train_recall_weighted = recall_score(all_labels_train, all_preds_train, average='weighted',zero_division=0)

        model.eval()
        val_loss, val_acc = 0, 0
        all_preds_val = []
        all_labels_val = []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                labels = labels.to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                outputs = model(input_ids, attention_mask)
                preds = outputs.argmax(dim=1)
                all_preds_val.extend(preds.cpu().numpy())  # 保存预测结果
                all_labels_val.extend(labels.cpu().numpy())  # 保存真实标签
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
        val_f1_macro = f1_score(all_labels_val, all_preds_val, average='macro',zero_division=0)
        val_f1_weighted = f1_score(all_labels_val, all_preds_val, average='weighted',zero_division=0)
        val_recall_macro = recall_score(all_labels_val, all_preds_val, average='macro',zero_division=0)
        val_recall_weighted = recall_score(all_labels_val, all_preds_val, average='weighted',zero_division=0)
        # 记录指标（关键！必须放在每个epoch循环内部）
        history['train_acc'].append(total_acc / len(train_data))
        history['train_f1_macro'].append(train_f1_macro)
        history['train_f1_weighted'].append(train_f1_weighted)
        history['train_recall_macro'].append(train_recall_macro)
        history['train_recall_weighted'].append(train_recall_weighted)

        history['val_acc'].append(val_acc / len(val_data))
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['val_recall_macro'].append(val_recall_macro)
        history['val_recall_weighted'].append(val_recall_weighted)
        print(f'''
                Epochs: {epoch + 1} 
                | Train Loss: {total_loss / len(train_data): .3f} 
                | Train Accuracy: {total_acc / len(train_data): .3f}
                | Train F1 (Macro): {train_f1_macro: .3f}
                | Train F1 (weighted): {train_f1_weighted: .3f}
                | Train Recall (Macro): {train_recall_macro: .3f}
                | Train Recall (weighted): {train_recall_weighted: .3f}
                | Val Loss: {val_loss / len(val_data): .3f} 
                | Val Accuracy: {val_acc / len(val_data): .3f}
                | Val F1 (Macro): {val_f1_macro: .3f}
                | Val F1 (weighted): {val_f1_weighted: .3f}
                | Val Recall (Macro): {val_recall_macro: .3f}
                | Val Recall (weighted): {val_recall_weighted: .3f}
                ''')
    plot_training_curves(history)
    return history
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=4)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    total_acc_test = 0
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            preds = output.argmax(dim=1)
            all_preds_test.extend(preds.cpu().numpy())  # 保存预测结果
            all_labels_test.extend(test_label.cpu().numpy())  # 保存真实标签
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    test_f1_macro = f1_score(all_labels_test, all_preds_test, average='macro',zero_division=0)
    test_f1_weighted = f1_score(all_labels_test, all_preds_test, average='weighted',zero_division=0)
    test_recall_macro = recall_score(all_labels_test, all_preds_test, average='macro',zero_division=0)
    test_recall_weighted = recall_score(all_labels_test, all_preds_test, average='weighted',zero_division=0)
    print(f'''
            Test Accuracy: {total_acc_test / len(test_data): .3f}
            Test F1 (Macro): {test_f1_macro: .3f}
            Test F1 (Weighted): {test_f1_weighted: .3f}
            Test Recall (Macro): {test_recall_macro: .3f}
            Test Recall (Weighted): {test_recall_weighted: .3f}
        ''')


import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_heatmap(model, tokenizer, sample_text, filename="attention_heatmap.png", head=0, max_length=50):
    # 编码输入文本
    inputs = tokenizer(sample_text,
                       padding='max_length',
                       truncation=True,
                       max_length=512,
                       return_tensors="pt")

    # 转移数据到设备
    device = next(model.parameters()).device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        _, attention_probs = model(input_ids, attention_mask, return_attention=True)

    # 处理注意力矩阵
    attention_matrix = attention_probs[0, head].cpu().numpy()  # 取第一个样本和指定头
    seq_len = attention_mask[0].sum().item()  # 有效序列长度
    attention_matrix = attention_matrix[:seq_len, :seq_len]  # 裁剪有效部分

    # 转换token为可读形式
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])
    tokens = [t.replace("Ġ", " ") for t in tokens]  # 处理RoBERTa空格符号

    # 截取最大显示长度
    if seq_len > max_length:
        attention_matrix = attention_matrix[:max_length, :max_length]
        tokens = tokens[:max_length]

    # 创建热力图
    plt.figure(figsize=(15, 15))
    sns.heatmap(attention_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="YlGnBu",
                linewidths=0.1,
                annot=False)

    # 调整标签显示
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title(f"Attention Head {head + 1}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
# model = RenyiBertClassifier(renyi_weights)
# EPOCHS = 10
# LR = 1e-6
# train_enhanced(model, df_train, df_val, LR, EPOCHS)
# evaluate(model, df_test)
# 训练和评估模型
model = RenyiBertClassifier(renyi_weights)
EPOCHS = 10
LR = 1e-5
train_enhanced(model, df_train, df_val, LR, EPOCHS)
evaluate(model, df_test)

# 可视化第一个测试样本的注意力
sample_text = df_test.iloc[0]['miniopcode']
plot_attention_heatmap(model, tokenizer, sample_text,
                      filename="attention_example.png",
                      head=3,  # 选择第4个头
                      max_length=30)  # 只显示前30个token
