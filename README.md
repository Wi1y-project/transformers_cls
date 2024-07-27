使用transformers + bert 实现的分类器

**数据需要放置在data下**  
-data  
 ---train.txt  
 ---dev.txt  
 ---test.txt  

**格式：(label\ttext)**  
eg: pos\t今天天气很好  
	neg\t不想吃饭  

train.py 训练
eval 评估
predict 预测

配置文件 config  
{  
    "bert_folder": "bert-base-uncased",  
    "checkpoint_dir": "./checkpoints",  
    "max_length": 128,  
    "batch_size": 32,  
    "num_labels": 10,  
    "learning_rate": 4e-5,  
    "epochs": 10,  
    "gradient_accumulation_steps": 1,  
    "warmup_steps": 100,  
    "weight_decay": 0.01,  
    "logging_dir": "./logs",  
    "logging_steps": 50,  
    "save_steps": 500,  
    "seed": 42,  
    "adam_epsilon": 1e-8,  
    "max_grad_norm": 1.0,  
    "early_stopping_patience": 10  
}  
