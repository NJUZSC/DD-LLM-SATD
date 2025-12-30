from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
import time  # 需导入 time 模块（若已导入可忽略）
import os
model1='DeepSeek-R1-Distill-Qwen-14B'

No_keywords='Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_MAT='Self-admitted technical debt (SATD) are technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, and XXX. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_Jitterbug='Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, and Workaround. Assign the label of SATD or Not-SATD for each given source code comment.'
Suggested_keywords_by_GPT4=' Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments. SATD comments usually contain specific keywords: TODO, FIXME, HACK, XXX, NOTE, DEBT, REFACTOR, OPTIMIZE, TEMP, WORKAROUND, KLUDGE, REVIEW, NOFIX, PENDING, and BUG. Assign the label of SATD or Not-SATD for each given source code comment.'


prompt_list = [No_keywords, Suggested_keywords_by_MAT, Suggested_keywords_by_Jitterbug, Suggested_keywords_by_GPT4]

# 3. 定义提示词对应的名称（用于输出文件命名，可选但推荐）
prompt_names = [
    "No_keywords",
    "Suggested_keywords_by_MAT",
    "Suggested_keywords_by_Jitterbug",
    "Suggested_keywords_by_GPT4"
]



# ====================== 1. 加载模型和分词器（修复分词器配置）======================
model_name = f"/home/zsc/llm_pre_trained_models/{model1}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 关键配置：Qwen 模型必须显式设置 pad_token 和 eos_token（确保生成终止）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 填充符复用结束符
tokenizer.padding_side = "left"  # 批量处理左填充更稳定
tokenizer.truncation_side = "right"  # 超长文本右侧截断（保留 prompt 核心指令）

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # 单卡推理（双卡可改为 "auto"，需安装 accelerate）
    trust_remote_code=True
).eval()  # 评估模式，禁用 Dropout

time_list = []
# ====================== 4. 批量推理函数（修复3处核心错误）======================
def batch_infer(comments, batch_size):
    start_timer = False  # 计时开关：是否开始计时
    timer = None  # 存储起始时间
    all_predictions = []
    for i in tqdm(range(0, len(comments), batch_size), desc="批量推理"):


        batch_comments = comments[i:i+batch_size]
        batch_messages = []
        
        # 1. 构建对话（Qwen 要求严格的 role 格式：system + user）
        for comment in batch_comments:
            user_content = PROMPT_TEMPLATE.format(comment=comment)
            # 关键：添加 system 指令约束模型行为（Qwen 必须通过 system 定义角色）
            messages = [
                {"role": "system", "content": "You are a classifier that only outputs YES or NO. Do not generate any other content."},
                {"role": "user", "content": user_content}
            ]
            batch_messages.append(messages)
        
        # 2. 格式化对话模板（保持 add_generation_prompt=True）
        batch_texts = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True  # 触发模型生成回复
        )
        
        # 3. 编码输入（修复：max_length 设为 512，避免输入被截断）
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Qwen2.5-3B 支持 8192，此处设 512 足够容纳 comment + 指令
        ).to(model.device)
        
        # 4. 批量生成（修复2处关键参数）
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,  # 仅生成 3 个 token（YES/NO 足够）
                do_sample=False,  # 确定性解码
                num_beams=1,      # 必须启用贪心搜索（之前注释掉了，导致生成异常）
                temperature=0.0,  # 无随机性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,  # 关键：指定结束符，强制生成终止
                repetition_penalty=1.1  # 可选：抑制重复生成
            )
        
        # 5. 解码后处理（保持不变）
        generated_ids_new = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        batch_responses = tokenizer.batch_decode(
            generated_ids_new,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 6. 提取 YES/NO
        for response in batch_responses:
            clean_response = response.strip().upper()
            #print(f"原始输出：{response} → 清洗后：{clean_response}")
            if "YES" in clean_response:
                all_predictions.append("YES")
            elif "NO" in clean_response:
                all_predictions.append("NO")
            else:
                all_predictions.append("UNKNOWN")
       
    
    return all_predictions



# 4. for 循环：用 enumerate() 获取 index 和 对应的 prompt_template
for idx, prompt_template in enumerate(prompt_list):
    # ====================== 2. 配置参数（修复关键错误）======================
    prompt_name=prompt_names[idx]
    INPUT_CSV_PATH = "/home/zsc/SATD_code/maldonado_corrected.csv"
    OUTPUT_CSV_PATH = f"/home/zsc/prompt_LLM/{model1}/infer_{prompt_name}.csv"
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    BATCH_SIZE = 16  # 3B模型24GB显存可稳定运行（不足则降为8）
    MAX_NEW_TOKENS = 5  # 仅需 YES/NO，最小化生成长度
    # 修复提示词：简洁明确，符合 Qwen 对话模型的指令理解习惯（避免冗余定义）
    PROMPT_TEMPLATE = prompt_template+''' Determine if the following code comment is SATD (Self-admitted technical debt). Answer ONLY YES or NO.
    Code comment: {comment}'''
    print(PROMPT_TEMPLATE)
    # ====================== 3. 读取 CSV 数据（保持不变）======================
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"原始数据总量：{len(df)}")


    if "comment_text" not in df.columns:
        raise ValueError("CSV 中未找到 'comment_text' 列，请检查列名")

    comment_list = df["comment_text"].dropna().tolist()
    valid_indices = df["comment_text"].notna().index.tolist()
    print(f"有效推理样本数：{len(comment_list)}")


    # ====================== 5. 执行推理并保存（保持不变）======================
    start_time = time.time()
    predictions = batch_infer(comment_list, BATCH_SIZE)
    # --- 在代码块结束后记录时间并计算差值 ---
    end_time = time.time()

    # 计算运行时间（秒）
    elapsed_time = end_time - start_time

    # 打印结果
    print(f"代码块运行总时间: {elapsed_time:.4f} 秒")
    df["predict"] = "UNKNOWN"
    df.loc[valid_indices, "predict"] = predictions

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")  # utf-8-sig 兼容中文和特殊字符

    print(f"\n推理完成！结果保存至：{OUTPUT_CSV_PATH}")
    print(f"生成结果分布：\n{df['predict'].value_counts()}")