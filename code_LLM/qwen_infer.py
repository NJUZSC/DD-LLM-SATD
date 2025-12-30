from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 指定模型名称并加载分词器与模型
model_name = "Qwen/Qwen2-7B-Chat"  # 也可尝试其他版本，如 "Qwen/Qwen2-1.5B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 确保分词器的填充侧设置为'left'，这对于批量处理时稳定性很重要 [1](@ref)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择合适的数据类型
    device_map="cuda:0",   # 自动分配至可用GPU
    trust_remote_code=True
).eval()  # 设置为评估模式

# 2. 准备输入：构建对话并应用模板
prompt = "请判断以下陈述是否正确：人工智能是研究如何制造智能机器的科学。请只回答 YES 或 NO。"
messages = [
    {"role": "user", "content": prompt}
]
# 将对话格式化为模型期望的模板 [1,5](@ref)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 添加提示符，告诉模型开始生成
)
# 将文本转换为模型可接受的张量格式
model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device) # 添加padding以确保输入格式统一 [1](@ref)

# 3. 文本生成 - 使用确定性解码确保输出稳定
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=5,        # 生成的最大token数量，限制输出长度
    do_sample=False,         # 关闭采样，使用确定性解码（贪心搜索）[5,6](@ref)
    num_beams=1,             # 束搜索的束宽为1，即贪心搜索，确保输出确定性 [6](@ref)
    temperature=0.0,         # 温度设为0，进一步消除随机性 [6](@ref)
    # 可选的：设置停止符，避免模型生成过多内容 [5](@ref)
    # eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id # 明确指定填充token
)

# 4. 解码并提取新生成的部分（跳过输入提示）[1,3](@ref)
generated_ids_new = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids_new, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

print("模型原始输出:", response)

# 5. 后处理：将输出标准化并提取YES/NO
output_text_clean = response.strip().upper()

# 通过判断输出中是否包含关键字符串来确定最终结果
if "YES" in output_text_clean:
    final_answer = "YES"
elif "NO" in output_text_clean:
    final_answer = "NO"
else:
    final_answer = "UNKNOWN"

print("最终判定结果:", final_answer)