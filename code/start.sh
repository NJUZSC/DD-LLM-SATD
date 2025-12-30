#!/bin/bash
#"apache-ant-1.7.0"
#"apache-jmeter-2.10"
    #"argouml"
   # "columba-1.4-src"
   #"jEdit-4.2"
    #"jfreechart-1.0.19"
   # "jruby-1.4.0"
   
    #"emf-2.4.1"
    #"hibernate-distribution-3.3.2.GA"
# 定义项目列表
PROJECTS=(
    "sql12"
)

# 固定参数
ACCUMULATION_STEPS=8         # finetune 脚本的梯度累积步数
USE_LORA="lora"             # 是否使用 LoRA
MODEL="Qwen2.5-3B-Instruct"

# 外层循环：遍历所有项目
for EXP_NAME in "${PROJECTS[@]}"; do
    echo "============================================"
    echo "=== Processing project: $EXP_NAME ==="
    echo "============================================"

    # 1. 执行 finetune 脚本
    echo "=== Running finetuning script ==="
    python /home/zsc/SATD_code/SATD_special_token_finetune.py \
        --exp_name "$EXP_NAME" \
        --accumulation_steps "$ACCUMULATION_STEPS" \
        --use_lora "$USE_LORA" \
        --model "$MODEL"

    # 检查上一步是否成功
    if [ $? -ne 0 ]; then
        echo "Finetuning for $EXP_NAME failed! Skipping to next project."
        continue  # 跳过当前项目的后续评估
    fi

    # 2. 依次执行 eval 脚本（epoch=1, 2, 3, 4, 5）
    #echo "=== Running evaluation scripts ==="
    #for epoch in 1 2 3 4 5; do
    #    echo "Evaluating epoch $epoch..."
    #    python /home/zsc/SATD_code/special_token_eval.py \
    #        --exp_name "$EXP_NAME" \
    #        --epoch "$epoch" \
    #        --use_lora "$USE_LORA" \
    #        --model "$MODEL"
        
        # 检查是否成功（失败仅记录，不终止）
    #    if [ $? -ne 0 ]; then
    #        echo "Evaluation for $EXP_NAME epoch $epoch failed!"
    #    fi
    #done
    # 2. 依次执行 eval 脚本（epoch=1, 2, 3, 4, 5）
    echo "=== Running evaluation scripts ==="
    python /home/zsc/SATD_code/special_token_eval.py \
        --exp_name "$EXP_NAME" \
        --epochs 15 \
        --use_lora "$USE_LORA" \
        --model "$MODEL"

    # 检查是否成功（失败仅记录，不终止）
    if [ $? -ne 0 ]; then
        echo "Evaluation for $EXP_NAME failed!"
    fi
done

echo "All projects and tasks completed."