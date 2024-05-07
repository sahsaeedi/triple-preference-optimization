<p align="center">
    <img alt="TPO" src="tpo_drawio.png" width="1000" height="">
</p>

<div align="center">
    
# TPO: Triple Preference Optimization
</div>

<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>

<!-- <a href="https://cogintlab-asu.github.io/" alt="asu">ASU</a> -->
<!-- <a href="https://www.microsoft.com/en-us/research/" alt="MSlogo"><img src="https://img.shields.io/badge/Microsoft-B1B479?logo=microsoft" /></a>
<a href="https://twitter.com/fe1ixxu">
  <img src="https://img.shields.io/twitter/follow/haoranxu?style=social&logo=twitter"
      alt="follow on Twitter"></a>
</p> -->


 **TPO** (**T**riple **P**reference **O**ptimization) is a new preference learning method designed to align an LLM with three preferences without requiring a separate supervised fine-tuning step. The simple one-step combination of SFT and Preference Optimization outperforms current state-of-the-art alignment methods such as DPO, CPO, KTO and IPO.

<p align="center">
    <img alt="TPO" src="demonstrations2.png" width="1000" height="">
</p>

## Contents 📄

- [Environment Setup](#environment-setup-)
- [Training with TPO](#training-with-tpo-)

## Supports ⭐

- **Models**: Compatible with various models including but not limited to `alignment-handbook/zephyr-7b-sft-full` and `mistralai/Mistral-7B-v0.1`.
- **GPUs**: Optimized for both Nvidia GPUs.
- **Batch Sizes**: Configurable batch sizes for training to optimize GPU usage.
- **Data Parallelism**: Support for data parallelism to speed up training on multiple GPUs.
<!-- - **Advanced Schedulers and Optimizers**: Includes support for cosine learning rate scheduling and AdamW optimization.
- **Precision Training**: Provides both mixed precision (bf16) and full precision training modes. -->

<!-- # Setup and Training 🚀 -->

## Environment Setup 🔧
This is a quick tutorial to set up and train a model with the TPO method.

1. **Create and activate a Conda environment**:
```bash
conda create --prefix tpo python=3.9 
conda activate tpo
```
2. **Install PyTorch with CUDA support (for Nvidia GPUs)**:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
3. **Install additional requirements**:
```bash
pip install -r requirements.txt
```
4. **Prepare the dataset**:
 In `./utils/dataset.py`, there is a parameter called `n.` You can change the size of the dataset by changing this parameter.
(We attached a sample dataset in the `data` folder we used for training. However, you can change the size.)
```bash
python dataset.py
```


## Training with TPO 🔥
Run  `run_tpo.sh` to train a model with TPO. This study used `alignment-handbook/zephyr-7b-sft-full` and `mistralai/Mistral-7B-v0.1` models. However, you can use other models to train with TPO.
```bash
#!/bin/bash

OUTPUT_DIR="/OUTPUT/DIR/PATH"

accelerate launch \
 --config_file configs/deepspeed_train_config_bf16.yaml \
  run_tpo.py  \
    --model_name_or_path mistralai/Mistral-7B-v0.1   \
    --tokenizer_name mistralai/Mistral-7B-v0.1   \
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
    --save_steps 100  \
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

```

