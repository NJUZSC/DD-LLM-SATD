import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

file_name = '/home/zsc/prompt_LLM/Qwen2.5-14B-Instruct/infer_Suggested_keywords_by_GPT4'

# 1. 读取CSV文件
df = pd.read_csv(f'{file_name}.csv')

# 新增：统计predict为UNKNOWN的行数和比例
unknown_count = df[df['predict'] == 'UNKNOWN'].shape[0]
total_rows = df.shape[0]
unknown_ratio = unknown_count / total_rows if total_rows > 0 else 0.0


# 2. 数据预处理
df['predict_binary'] = df['predict'].apply(lambda x: 1 if x == 'YES' else 0)
df['satd'] = df['satd'].astype(int)

# 3. 按照project_name分组计算指标
projects_metrics = []

# 获取所有唯一的project_name
project_names = df['project_name'].unique()

for project in project_names:
    # 获取当前项目的子数据集
    project_df = df[df['project_name'] == project]
    
    true_labels = project_df['satd'].tolist()
    predicted_labels = project_df['predict_binary'].tolist()
    
    # 计算当前项目的TP, FP, TN, FN
    TP = FP = TN = FN = 0
    
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 1 and pred == 0:
            FN += 1
    
    # 计算各项指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    # 使用sklearn计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 存储当前项目的指标
    project_metrics = {
        'project_name': project,
        'total_samples': len(project_df),
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    projects_metrics.append(project_metrics)

# 4. 计算所有项目的平均指标
avg_metrics = {
    'precision': np.mean([pm['precision'] for pm in projects_metrics]),
    'recall': np.mean([pm['recall'] for pm in projects_metrics]),
    'f1': np.mean([pm['f1'] for pm in projects_metrics]),
    'accuracy': np.mean([pm['accuracy'] for pm in projects_metrics])
}

# 5. 输出结果到控制台和文件
output_filename = f"{file_name}_by_project.txt"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write("模型分类评估报告（按项目分组）\n")
    f.write("=" * 60 + "\n\n")
    
    # 输出每个项目的详细指标
    for i, pm in enumerate(projects_metrics, 1):
        f.write(f"项目 {i}: {pm['project_name']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"总样本数: {pm['total_samples']}\n")
        f.write(f"混淆矩阵:\n")
        f.write(f"[[TN: {pm['TN']:3d}  FP: {pm['FP']:3d}]\n")
        f.write(f" [FN: {pm['FN']:3d}  TP: {pm['TP']:3d}]]\n")
        f.write(f"Precision (精确率): {pm['precision']:.4f}\n")
        f.write(f"Recall (召回率):   {pm['recall']:.4f}\n")
        f.write(f"F1 Score:          {pm['f1']:.4f}\n")
        f.write("\n")
    
    # 输出平均指标
    f.write("=" * 60 + "\n")
    f.write("所有项目平均指标\n")
    f.write("=" * 60 + "\n")
    f.write(f"平均 Precision: {avg_metrics['precision']:.4f}\n")
    f.write(f"平均 Recall:    {avg_metrics['recall']:.4f}\n")
    f.write(f"平均 F1 Score:  {avg_metrics['f1']:.4f}\n")
     # 新增：添加UNKNOWN统计信息
    f.write(f"UNKNOWN 行数:    {unknown_count:d}\n")
    f.write(f"UNKNOWN 比例:    {unknown_ratio:.4f} ({unknown_ratio*100:.2f}%)\n")
    f.write("\n")
    
    f.write("指标解释:\n")
    f.write("- Precision: 模型预测为YES的样本中，真实为YES的比例。值越高，误报越少。\n")
    f.write("- Recall:    所有真实为YES的样本中，被模型成功找出的比例。值越高，漏报越少。\n")
    f.write("- F1 Score:  Precision和Recall的调和平均数，综合评估模型性能。\n")

# 6. 同时在控制台输出汇总信息
print("=" * 60)
print("模型评估结果（按项目分组）")
print("=" * 60)

for i, pm in enumerate(projects_metrics, 1):
    print(f"\n项目 {i}: {pm['project_name']}")
    print(f"总样本数: {pm['total_samples']}")
    print(f"Precision: {pm['precision']:.4f}, Recall: {pm['recall']:.4f}, F1: {pm['f1']:.4f}")

print("\n" + "=" * 60)
print("所有项目平均指标")
print("=" * 60)
print(f"平均 Precision: {avg_metrics['precision']:.4f}")
print(f"平均 Recall:    {avg_metrics['recall']:.4f}")
print(f"平均 F1 Score:  {avg_metrics['f1']:.4f}")
print("=" * 60)
print(f"详细结果已写入文件: {output_filename}")