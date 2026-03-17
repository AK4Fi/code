from transformers import RobertaTokenizer, RobertaModel
import torch

def chunk_sequence(opcodes, window_size=510, stride=255):
    """滑动窗口分块（保留上下文）"""
    chunks = []
    for i in range(0, len(opcodes), stride):
        chunk = opcodes[i:i+window_size]
        if len(chunk) < window_size:
            chunk += ['[PAD]'] * (window_size - len(chunk))  # 填充短块
        chunks.append(chunk)
    return chunks

# 加载本地CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("E:\pycharmcode\codebert-base")
model = RobertaModel.from_pretrained("E:\pycharmcode\codebert-base")

# 加载操作码序列
with open("0B2RwKm6dq9fjUWDNIOa.txt", "r") as f:
    opcodes = f.read().split()

# 分块处理
chunks = chunk_sequence(opcodes, window_size=510, stride=255)

# 逐块编码并聚合特征
all_features = []
for chunk in chunks:
    inputs = tokenizer(
        chunk,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        is_split_into_words=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    all_features.append(outputs.last_hidden_state[:, 0, :])  # 取[CLS]向量

# 特征聚合（均值池化）
combined_feature = torch.mean(torch.stack(all_features), dim=0)
print("聚合后特征维度:", combined_feature.shape)  # torch.Size([1, 768])