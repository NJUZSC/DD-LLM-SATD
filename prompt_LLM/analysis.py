import pandas as pd
import numpy as np

# 文件路径
file_name = '/home/zsc/prompt_LLM/Qwen2.5-3B-Instruct/infer_Suggested_keywords_by_MAT'

# 1. 读取CSV文件
df = pd.read_csv(f'{file_name}.csv')

# 2. 定义token统计函数（默认按空格分割，可根据实际token规则调整）
def calculate_token_stats(text_series):
    """
    计算文本序列的token统计指标：平均数、中位数、最小值、最大值
    """
    # 计算每行的token数量（空文本记为0）
    token_counts = text_series.fillna('').apply(lambda x: len(str(x).split()))
    # 计算统计值
    stats = {
        '平均数': np.mean(token_counts),
        '中位数': np.median(token_counts),
        '最小值': np.min(token_counts),
        '最大值': np.max(token_counts)
    }
    return stats, token_counts

# 3. 筛选两类数据
# 3.1 predict列为UNKNOWN的行
df_unknown = df[df['predict'] == 'UNKNOWN']
# 3.2 predict列不为UNKNOWN的行
df_not_unknown = df[df['predict'] != 'UNKNOWN']

# 4. 计算两类数据的token统计
unknown_stats, unknown_token_counts = calculate_token_stats(df_unknown['comment_text'])
not_unknown_stats, not_unknown_token_counts = calculate_token_stats(df_not_unknown['comment_text'])

# 5. 将结果写入txt文件
output_path = f'{file_name}_token_stats.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('=== Predict列为UNKNOWN的comment_text token统计 ===\n')
    for key, value in unknown_stats.items():
        f.write(f'{key}: {value:.2f}\n')  # 保留两位小数
    f.write('\n=== Predict列不为UNKNOWN的comment_text token统计 ===\n')
    for key, value in not_unknown_stats.items():
        f.write(f'{key}: {value:.2f}\n')

print(f'统计结果已写入：{output_path}')
# 可选：打印部分结果验证
print(f"\nUNKNOWN类token统计：{unknown_stats}")
print(f"非UNKNOWN类token统计：{not_unknown_stats}")