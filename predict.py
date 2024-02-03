import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json

# 加载配置文件
with open('config.json') as f:
    config = json.load(f)

# 设置device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(config['bert_folder'])
model = BertForSequenceClassification.from_pretrained(config['checkpoint_dir'])
model.to(device)

# 加载标签映射
with open('data/labels.txt', 'r') as f:
    labels = f.read().splitlines()
label_map = {i: label for i, label in enumerate(labels)}

# 预测函数
def predict(text):
    model.eval()
    predictions = []

    with torch.no_grad():

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config['max_length'])
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

    return preds.item()

# 读取文本数据
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts]

# 主程序
def main(predict_file):
    texts = load_text(predict_file)
    pred_labels = predict(texts)
    for text, label in zip(texts, pred_labels):
        print(f"Text: {text}\nPredicted Label: {label}\n")

if __name__ == "__main__":

    text = "Physician judgment about malnutrition and risk factors for malnutrition were also evaluated."
    res = predict(text)
    print(res)