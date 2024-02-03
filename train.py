import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os
from tqdm import tqdm, trange
import json

# Load configuration parameters
with open('config.json') as f:
    config = json.load(f)

labels_dict = {}

with open('data/labels.txt', 'r', encoding='utf-8') as f:
    for c, each in enumerate(f):
        each = each.strip()
        labels_dict[each] = c

# 设置随机种子，以保证实验的可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config['seed'])

# 加载预训练的tokenizer和BERT模型
tokenizer = BertTokenizer.from_pretrained(config['bert_folder'])
model = BertForSequenceClassification.from_pretrained(config['bert_folder'], num_labels=config['num_labels'])

# 确定设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载数据
def load_data(filename):
    labels = []
    texts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label, text = line.strip().split('\t')
            labels.append(labels_dict[label])
            texts.append(text)
    return texts, labels

train_texts, train_labels = load_data('data/train_data.txt')
dev_texts, dev_labels = load_data('data/test_data.txt')

# 编码数据
def encode_data(texts, labels):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config['max_length'],
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_masks, labels)

train_dataset = encode_data(train_texts, train_labels)
dev_dataset = encode_data(dev_texts, dev_labels)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config['batch_size'])
dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=config['batch_size'])

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
total_steps = len(train_dataloader) * config['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)

# 训练模型
def train_model(epochs, model, train_dataloader, dev_dataloader):
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs, desc="Epoch")

    # 设置早停参数
    best_accuracy = 0
    early_stopping_counter = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # 评估模型性能
            if config['logging_steps'] > 0 and global_step % config['logging_steps'] == 0:
                dev_loss, dev_accuracy = evaluate_model(model, dev_dataloader)
                print(f"\nEvaluation loss: {dev_loss},accuracy: {dev_accuracy}")

                # 保存最佳模型
                if dev_accuracy > best_accuracy:
                    best_accuracy = dev_accuracy
                    early_stopping_counter = 0
                    output_dir = os.path.join(config['checkpoint_dir'], 'best_model')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                else:
                    early_stopping_counter += 1

                # 早停检查
                if early_stopping_counter >= config['early_stopping_patience']:
                    print("\nEarly stopping triggered.")
                    return global_step, tr_loss / global_step

    return global_step, tr_loss / global_step

# 评估函数
def evaluate_model(model, validation_dataloader):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(out_label_ids, preds)
    return eval_loss, accuracy

# 开始训练
global_step, train_loss = train_model(config['epochs'], model, train_dataloader, dev_dataloader)
print("Training finished.")

# 保存最终模型
output_dir = os.path.join(config['checkpoint_dir'], 'final_model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)