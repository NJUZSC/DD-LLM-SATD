import sys
import traceback
import os
import random
import copy
import csv
import pandas as pd
import torch
import numpy as np
import math
import re
import ast
from transformers import AutoTokenizer
from tqdm import tqdm
class SATDDataset(torch.utils.data.IterableDataset):
    def __init__(self, exp_name, global_rank, world_size, tokenizer, max_len, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.df = pd.read_csv('/home/zsc/SATD_code/maldonado_corrected.csv')
            self.train_csv = self.df[self.df['project_name'] == exp_name]

            rows,_ = self.train_csv.shape
            part_len = rows // world_size
            print(f">>> world size is: {world_size}, parquet file nums is: {rows}, file nums for per process is: {part_len}")
            self.startIndex = global_rank * part_len
            self.endIndex = self.startIndex + part_len        
        self.global_rank = global_rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.max_len=max_len
        self.task=exp_name
 
        print('=======>>>>> init note pair dataset', global_rank, world_size)
        print("get in process init() 1")

    def __len__(self):
        rows,_ = self.train_csv.shape
        return int(rows)

    def generate_a_conversation(self,text):    
        system_prompt = """You are an expert in identifying Self-Admitted Technical Debt (SATD) in code. 
        SATD is explicitly marked by developers through:
        1. Tags: TODO/FIXME/HACK/XXX
        2. Keywords: "temporary", "workaround", "fix later"
        3. Context: Admissions of suboptimal solutions or known issues"""      
        comments=f"This is a code comment that needs to be judged to determine whether it is SATD: {text}."
        embedding_ques = "Please generate a token to uniquely represent the information of this code comment."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'{comments}{embedding_ques}'},
            {"role": "assistant", "content": '<|embedding|>'}

        ]
        
        return messages
        

    def prepare_batchdata_for_train(self, batch_data):
        # 遍历 batch_infos
        batch_convs=[]
        batch_labels=[]
        batch_comments=[]
        for item in batch_data:
            # 获取 comment_text 和 satd 的值
            comment_text = item['comment_text']
            satd = item['satd']
            conv=self.generate_a_conversation(comment_text)
            batch_convs.append(conv)
            batch_labels.append(satd)
            batch_comments.append(comment_text)
        batch_texts = [self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in batch_convs]
        model_inputs = self.tokenizer(batch_texts,padding=True, return_tensors="pt")
        ret = {
            'inputs': model_inputs,
            'labels':torch.Tensor(batch_labels),
            'comments': batch_comments
        }
        return ret
    
    
    def _sample_generator(self, start_index, end_index, worker_id):
        
        batch_size = 1
        selected_files = self.train_csv[start_index:end_index]

        #print(f'producer process: {worker_id} {start_index} {end_index}')
        # while True:
        selected_files = selected_files.sample(frac=1).reset_index(drop=True)
        print('after shuffle:\n',selected_files)
        rows,cols = selected_files.shape
        
        for row in range(0,rows,batch_size):
            batch_infos = []
            batch_size = min(batch_size,rows - row)
            # 遍历当前 batch 的每一行
            for i in range(batch_size):
                # 获取当前行的数据
                infos = selected_files.iloc[row + i]
                
                # 将当前行转换为字典格式，并添加到 batch_infos 中
                batch_infos.append({
                    'comment_text': infos['comment_text'],
                    'satd': int(infos['satd'])
                })
            #process batch data to token
            batch_data = copy.deepcopy(batch_infos)  
            try:
                train_batch_data = self.prepare_batchdata_for_train(batch_data)

                yield train_batch_data
            except Exception as e:
                print("Error:", str(e))
            finally:
                batch_data = []
                
        
        

    def __iter__(self):
        if self.mode == 'train':
            worker_info = torch.utils.data.get_worker_info()
            worker_id = None
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = self.startIndex
                iter_end = self.endIndex
                print("1.单线程进入")
            else:  # in a worker process
                per_worker = int(math.ceil((self.endIndex - self.startIndex) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = self.startIndex + worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.endIndex)
                print("多线程：",iter_start,iter_end)
            sampler_iterator = self._sample_generator(iter_start, iter_end, worker_id)
            return sampler_iterator
        




if __name__ == "__main__":
    #config_path = '/mnt/nj-public02/usr/xiangyiwei/minicpmv-train-embedding-dev_v3_benchmark/config/train_mllm_data_sft_aligned_cpm.yaml'
    model_dir = '/home/zsc/llm_pre_trained_models/Qwen2.5-1.5B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 定义新的特殊 token
    new_special_tokens = ["<|SATD_YES|>", "<|SATD_NO|>","<|embedding|>"]
    
    # 添加新的特殊 token
    num_added_toks = tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    print(f"Added {num_added_toks} tokens")
    exp_name='apache-ant-1.7.0'
    train_dataset = SATDDataset(exp_name, 0, 1, tokenizer, 2048, mode='train')
    satd_embedding_id = tokenizer.convert_tokens_to_ids("<|embedding|>")
    print(f"<embedding> id: {satd_embedding_id}")
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None, batch_sampler=None, num_workers=0, shuffle=False)
    print(len(dataloader))
    for batch in tqdm(dataloader, desc="Processing Batches"):
        print(batch)
        break
        #import pdb;pdb.set_trace()

    print('done')
