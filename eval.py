import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import classification_report
import numpy as np
import json

# Load configuration parameters
with open('config.json') as f:
    config = json.load(f)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(config['bert_folder'])
model = BertForSequenceClassification.from_pretrained(config['checkpoint_dir']+'/final_model')
model.to(device)

# 加载标签
with open('data/labels.txt', 'r') as f:
    labels = f.read().splitlines()
label_map = {i: label for i, label in enumerate(labels)}

labels_dict = {}

with open('data/labels.txt', 'r', encoding='utf-8') as f:
    for c, each in enumerate(f):
        each = each.strip()
        labels_dict[each] = c

# 加载和编码数据的函数
def load_and_encode_data(tokenizer, file_path, max_length):
    texts, true_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label, text = line.strip().split('\t')
            texts.append(text)
            true_labels.append(label_map[labels_dict[label]])

    print([label_map[labels_dict[label]] for label in true_labels])
    encoded_batch = tokenizer(texts, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True,
                              max_length=max_length, return_tensors='pt')

    dataset = TensorDataset(encoded_batch['input_ids'], encoded_batch['attention_mask'],
                            torch.tensor([labels_dict[label] for label in true_labels]))
    return dataset


# 评估函数
def evaluate_model(model, dataset):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config['batch_size'])

    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, axis=-1)

        predictions.extend(preds.detach().cpu().numpy())
        true_labels.extend(batch[2].detach().cpu().numpy())

    return true_labels, predictions


# 加载数据
dev_dataset = load_and_encode_data(tokenizer, 'data/test_data.txt', config['max_length'])

# 对模型进行评估
true_labels, predictions = evaluate_model(model, dev_dataset)

# 输出分类报告
report = classification_report(true_labels, predictions, target_names=labels, digits=4)
print(report)