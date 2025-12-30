import os
import torch
import json
import os
from tqdm import tqdm
import torch.nn as nn
import shutil
from SATD_special_token_Qwen2_5_modeling_2_classifier import SATDAutoModel
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from peft import PeftModel
from SATD_special_token_dataset_for_eval import SATDDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix

from peft import LoraConfig, get_peft_model

device = "cuda"
import time



def train(training_args):
    best_f1=0
    
    if training_args.use_lora=="lora":
        model_dir = f'/home/zsc/llm_pre_trained_models/{training_args.model}'
        model = SATDAutoModel.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto"
        )
        lora_weights_path = f'/home/zsc/SATD_code/RQ123/apache-ant-1.7.0/lora_Qwen2.5-3B-Instruct_finetune_apache-ant-1.7.0_4'
        tokenizer = AutoTokenizer.from_pretrained(lora_weights_path)
        # 调整模型的 token embeddings 大小
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, lora_weights_path)
        # 查看特殊 token 的编码
        satd_yes_id = tokenizer.convert_tokens_to_ids("<|SATD_YES|>")
        satd_no_id = tokenizer.convert_tokens_to_ids("<|SATD_NO|>")
        satd_embedding_id = tokenizer.convert_tokens_to_ids("<|embedding|>")
    else:
        model_dir = f'/home/zsc/SATD_code/{training_args.exp_name}/{training_args.use_lora}_{training_args.model}_finetune_{training_args.exp_name}_{epoch}'
        model = SATDAutoModel.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        satd_yes_id = tokenizer.convert_tokens_to_ids("<|SATD_YES|>")
        satd_no_id = tokenizer.convert_tokens_to_ids("<|SATD_NO|>")
        satd_embedding_id = tokenizer.convert_tokens_to_ids("<|embedding|>")

    print(f"<|SATD_YES|> id: {satd_yes_id}")
    print(f"<|SATD_NO|> id: {satd_no_id}")
    print(f"<|embedding|> id: {satd_embedding_id}")
    
    
    
    train_dataset = SATDDataset(training_args.exp_name, 0, 1, tokenizer, 2048, mode='train')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=0
    )

    # 初始化列表
    comment_lists = []
    pred_lists = []
    label_lists = []
    model.eval()
    # --- 在代码块开始前记录时间 ---
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Processing Batches"):

            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device).long()  # 将 labels 转换为 Long 类型，并移动到 device
            l_outputs = model(**inputs)
            # 获取预测结果
            preds = torch.argmax(l_outputs, dim=-1)  # 对 logits 进行 argmax 得到预测结果
            
            # 将 preds 和 labels 添加到列表中
            comment_lists.extend(batch['comments'])
            pred_lists.extend(preds.cpu().numpy().tolist())
            label_lists.extend(labels.cpu().numpy().tolist())
    # --- 在代码块结束后记录时间并计算差值 ---
    end_time = time.time()

    # 计算运行时间（秒）
    elapsed_time = end_time - start_time

    # 打印结果
    print(f"代码块运行总时间: {elapsed_time:.4f} 秒")
    # 初始化统计变量
    TP = 0  # True Positive
    FP = 0  # False Positive
    TN = 0  # True Negative
    FN = 0  # False Negative

    # 遍历每个预测和真实标签
    for pred, label in zip(pred_lists, label_lists):
        if label == 1 and pred == 1:
            TP += 1
        elif label == 0 and pred == 1:
            FP += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 1 and pred == 0:
            FN += 1

    # 计算 Precision、Recall、F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 打印结果
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
        

    
if __name__ == "__main__":
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='这是一个示例程序。')

    # 添加参数
    parser.add_argument('--save_freq', default=20000, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accumulation_steps', default=8, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--exp_name', default='apache-ant-1.7.0', type=str)
    parser.add_argument('--use_lora', default="lora", type=str)
    parser.add_argument('--model', default="Qwen2.5-3B-Instruct", type=str)


    # 解析命令行参数
    training_args = parser.parse_args()

    train(training_args)
    
