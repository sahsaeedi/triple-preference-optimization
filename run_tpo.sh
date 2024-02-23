#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export TRANSFORMERS_CACHE= .cache/mistral-7b
OUTPUT_DIR=${1:-"/TO/SAVE/THR/MODEL"}

python ./run_tpo.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --tokenizer_name mistralai/Mistral-7B-v0.1 \
    --beta 0.1 \
    --alpha 0.5 \
    --do_train \
    --bf16 \
    --multi_gpu_one_model True \
    --learning_rate 5.0e-7 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --log_level info \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --seed 42 \
    --overwrite_output_dir \
    --report_to none
