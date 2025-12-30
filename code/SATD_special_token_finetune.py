import os
import torch
import json
import os
from tqdm import tqdm
import torch.nn as nn

from SATD_special_token_Qwen2_5_modeling_2_classifier import SATDAutoModel
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

from SATD_special_token_dataset import SATDDataset
import matplotlib.pyplot as plt


from peft import LoraConfig, get_peft_model

device = "cuda"


def train(training_args):
    model_dir = f'llm_pre_trained_models/{training_args.model}'
    model = SATDAutoModel.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # special token
    new_special_tokens = ["<|SATD_YES|>", "<|SATD_NO|>","<|embedding|>"]

    # special token
    num_added_toks = tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    print(f"Added {num_added_toks} tokens")


    # resize token embeddings 
    model.resize_token_embeddings(len(tokenizer))
    
    
    satd_yes_id = tokenizer.convert_tokens_to_ids("<|SATD_YES|>")
    satd_no_id = tokenizer.convert_tokens_to_ids("<|SATD_NO|>")
    satd_embedding_id = tokenizer.convert_tokens_to_ids("<|embedding|>")

    print(f"<|SATD_YES|> id: {satd_yes_id}")
    print(f"<|SATD_NO|> id: {satd_no_id}")
    print(f"<|embedding|> id: {satd_embedding_id}")
    
    if training_args.use_lora=="lora":
        lora_config = LoraConfig(
            modules_to_save = ['Classifier'],
            r=16,  # LoRA r
            lora_alpha=128,  # LoRA alpha
            lora_dropout=0.05,  # LoRA dropout
            target_modules=["q_proj", "v_proj", "k_proj","o_proj"]  # attention
        )
        model = get_peft_model(model, lora_config)
    print(model)
   
    
    
    
    train_dataset = SATDDataset(training_args.exp_name, 0, 1, tokenizer, 2048, mode='train')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=0
    )


    epochs = 10
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-6)
    steps = 0
    loss_values = []
    accumulated_avg_loss = 0
    # import pdb
    # pdb.set_trace()
    for epoch in range(epochs): 
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for batch in tqdm(train_loader, desc="Processing Batches"):
            
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device).long()  
            
            if training_args.use_lora !="lora":
                l_outputs = model(**inputs)
                #r_outputs = self.model(**r_inputs)
            else:
                with model._enable_peft_forward_hooks(**inputs):
                    l_outputs = model.base_model(**inputs)
            # loss
        
            loss = F.cross_entropy(l_outputs, labels)

            loss = loss / training_args.accumulation_steps
            accumulated_avg_loss += loss.item()
            loss.backward()
            steps=steps+1
            if steps % training_args.accumulation_steps == 0:
                print(f"Epoch {epoch+1}, Step {steps}, Loss: {accumulated_avg_loss}")
                loss_values.append(accumulated_avg_loss)
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
        os.makedirs(f'/SATD_code/{training_args.exp_name}', exist_ok=True)
        os.makedirs(f'/SATD_code/{training_args.exp_name}/{training_args.use_lora}_{training_args.model}_finetune_{training_args.exp_name}_{epoch+1}', exist_ok=True)
        model.save_pretrained(f'/SATD_code/{training_args.exp_name}/{training_args.use_lora}_{training_args.model}_finetune_{training_args.exp_name}_{epoch+1}')
        tokenizer.save_pretrained(f'/SATD_code/{training_args.exp_name}/{training_args.use_lora}_{training_args.model}_finetune_{training_args.exp_name}_{epoch+1}')

    plt.plot(range(1, len(loss_values) + 1), loss_values, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Steps")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'/SATD_code/{training_args.exp_name}/embedding_loss.png')
    
if __name__ == "__main__":
    import argparse

    #  ArgumentParser 对象
    parser = argparse.ArgumentParser(description='这是一个示例程序。')

    # parm
    parser.add_argument('--save_freq', default=20000, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--accumulation_steps', default=8, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--exp_name', default='apache-ant-1.7.0', type=str)
    parser.add_argument('--use_lora', default="full", type=str)
    parser.add_argument('--model', default="Qwen2.5-0.5B-Instruct", type=str)


    # 解析命令行参数
    training_args = parser.parse_args()

    train(training_args)
    
