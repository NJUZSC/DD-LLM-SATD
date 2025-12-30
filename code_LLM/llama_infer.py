import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM  # 补充缺失导入
import torch
import pandas as pd
from tqdm import tqdm
import time
import os

# 1. ====================== 模型配置 =======================
# 本地模型路径或 Hugging Face 模型 ID（确保有访问权限）
model_path = '/home/zsc/llm_pre_trained_models/Meta-Llama-3.1-8B-Instruct'
# 提取模型名称（用于输出路径，避免路径冗余）
model_name = os.path.basename(model_path)

# 提示词列表（移除重复的任务描述，统一在 build_llama3_prompt 中控制输出格式）
No_keywords='Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_MAT='Self-admitted technical debt (SATD) are technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, and XXX. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_Jitterbug='Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, and Workaround. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_GPT4=' Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, XXX, NOTE, DEBT, REFACTOR, OPTIMIZE, TEMP, WORKAROUND, KLUDGE, REVIEW, NOFIX, PENDING, and BUG. Assign the label of SATD or Not-SATD for each given source code comment.'

prompt_list = [No_keywords, Suggested_keywords_by_MAT, Suggested_keywords_by_Jitterbug, Suggested_keywords_by_GPT4]
prompt_names = [
    "No_keywords",
    "Suggested_keywords_by_MAT",
    "Suggested_keywords_by_Jitterbug",
    "Suggested_keywords_by_GPT4"
]

# 2. ====================== 加载模型和分词器（Llama 3 适配）======================
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 确保 padding token 存在（Llama 模型默认没有 pad token，用 eos token 替代）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 生成时 padding_side 设为 left（避免影响生成结果，可选但推荐）
tokenizer.padding_side = "left"

# 加载模型（bfloat16 更高效，device_map="auto" 自动分配 GPU/CPU）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True  # 减少 CPU 内存占用
).eval()  # 推理模式，禁用 Dropout

# 3. ====================== Prompt 构建函数（核心优化）======================
def build_llama3_prompt(system_instruction, comment):
    """
    严格遵循 Llama 3 Instruct 官方模板：
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    系统指令<|eot_id|><|start_header_id|>user<|end_header_id|>
    用户输入（评论）<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_instruction}
Your task: Determine if the comment is SATD. Answer ONLY with "SATD" or "Not-SATD". Do not add any explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Comment: {comment}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# 4. ====================== 批量推理函数（优化输出映射）======================
def batch_infer(comments, batch_size, system_instruction):
    all_predictions = []
    for i in tqdm(range(0, len(comments), batch_size), desc="批量推理"):
        batch_comments = comments[i:i+batch_size]
        # 构建批量 Prompt
        batch_texts = [build_llama3_prompt(system_instruction, comment) for comment in batch_comments]
        
        # 编码（truncation=True 避免超长 Prompt）
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 生成（禁用梯度计算，提升速度）
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10,  # 只需要输出 "SATD" 或 "Not-SATD"，10 个 token 足够
                do_sample=False,  # 确定性生成，避免随机
                temperature=0.0,  # 温度为 0，完全确定
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # 轻微抑制重复
            )
        
        # 解码（只取生成的部分，忽略输入 Prompt）
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        batch_responses = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 映射输出（容错性更强）
        for response in batch_responses:
            clean_resp = response.strip().lower()
            if "satd" in clean_resp and "not" not in clean_resp:
                all_predictions.append("SATD")
            elif "not" in clean_resp or "no" in clean_resp:
                all_predictions.append("Not-SATD")
            else:
                all_predictions.append("UNKNOWN")
    
    return all_predictions

# 5. ====================== 主循环（修正路径和 Prompt 逻辑）======================
BATCH_SIZE = 16  # 根据 GPU 显存调整（8B 模型建议 16-32）
INPUT_CSV_PATH = "/home/zsc/SATD_code/maldonado_corrected.csv"
OUTPUT_ROOT = f"/home/zsc/prompt_LLM/{model_name}"  # 输出根路径（用模型名称作为子目录）
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for idx, system_instruction in enumerate(prompt_list):
    prompt_name = prompt_names[idx]
    output_csv = os.path.join(OUTPUT_ROOT, f"infer_{prompt_name}.csv")
    
    print(f"\n{'='*50}")
    print(f"Testing prompt: {prompt_name}")
    print(f"Output path: {output_csv}")
    
    # 读取数据
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"原始数据总量：{len(df)}")
    df=df[:1000]
    
    if "comment_text" not in df.columns:
        raise ValueError("CSV 中未找到 'comment_text' 列，请检查列名")
    
    # 过滤空评论
    comment_list = df["comment_text"].dropna().tolist()
    valid_indices = df["comment_text"].notna().index.tolist()
    print(f"有效推理样本数：{len(comment_list)}")
    
    # 执行推理
    start_time = time.time()
    predictions = batch_infer(comment_list, BATCH_SIZE, system_instruction)
    elapsed_time = time.time() - start_time
    
    # 保存结果
    df["predict"] = "UNKNOWN"
    df.loc[valid_indices, "predict"] = predictions
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    # 打印统计信息
    print(f"推理耗时：{elapsed_time:.2f} 秒（{len(comment_list)/elapsed_time:.2f} 样本/秒）")
    print(f"结果分布：\n{df['predict'].value_counts()}")
    print(f"{'='*50}")