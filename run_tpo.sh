#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# export HF_HOME= .cache/mistral-7b
OUTPUT_DIR="/"

accelerate launch \
 --config_file configs/deepspeed_train_config_bf16.yaml \
  run_tpo.py  \
    --model_name_or_path microsoft/phi-2   \
    --tokenizer_name microsoft/phi-2   \
    --beta 0.1  \
    --alpha 0.5  \
    --do_train  \
    --bf16   \
    --attn_implementation flash_attention_2 \
    --multi_gpu_one_model True  \
    --learning_rate 5.0e-7 \
    --gradient_accumulation_steps 2  \
    --lr_scheduler_type cosine  \
    --optim adamw_torch  \
    --warmup_ratio 0.1   \
    --save_steps 10  \
    --log_level info   \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  \
    --evaluation_strategy steps   \
    --save_total_limit 1  \
    --logging_strategy steps \
    --logging_steps 10   \
    --output_dir $OUTPUT_DIR  \
    --num_train_epochs 1  \
    --max_length 1024   \
    --max_prompt_length 512 \
    --seed 42  \
    --overwrite_output_dir \
    --report_to none
