# DD-LLM-SATD
This repository is a replica package of the paper "DD-LLM SATD:A Cost-Effective and Efficient Discriminative Approach to SATD Identification Using Decoder-Only LLMs", including the implementation code of DD-LLM SATD, the preprocessed complete dataset used for training, and infer.

This repository is organized into the following four folders:

#### `code`
Stores the complete training and testing scripts for our model. This includes data preprocessing, model training loops, evaluation metrics, and inference scripts.
#### `code_LLM`
Inference code for three representative Instruct Large Language Models: DeepSeek, Llama and Qwen
#### `dataset`
Contains the datasets required for the experiments. The datasets are split into training, validation, and test sets as described in the paper.
#### `pre-trained_model`
Stores the pre-trained model checkpoint used as the starting point for our fine-tuning process.
#### `prompt_LLM`
It contains the relevant implementation code and corresponding results for RQ2.

#### `result_sota`
Stores the SOTA experimental results of our model, including detailed logs, performance metrics across different runs, and visualizations.

==============================================================
### Code Folder Structure
- `code/`
  - `SATD_special_token_dataset_for_eval.py` (The test dataset and prompt is provided here.)
  - `SATD_special_token_dataset.py` (The train dataset and prompt is provided here.)
  - `SATD_special_token_finetune.py` (The train code is provided here.)
  - `SATD_special_token_Owen2_5_modeling_2_classifier.py` (The train model is provided here.)
  - `special_token_eval.py` (The test code is provided here.)
  - `start.sh` (The train and test script is provided here.)
==============================================================
### Training Pipeline
To train the model, you need to first download Qwen2.5 pre-trained models with various parameter counts from the Hugging Face official website and place them in the pre_train_model folder.

Then execute bash start.sh. Several parameters are available here:
```
python /code/SATD_special_token_finetune.py \
        --exp_name "$EXP_NAME" \
        --accumulation_steps "$ACCUMULATION_STEPS" \
        --use_lora "$USE_LORA" \
        --model "$MODEL"
```

```
python /code/special_token_eval.py \
        --exp_name "$EXP_NAME" \
        --epochs 15 \
        --use_lora "$USE_LORA" \
        --model "$MODEL"
```

