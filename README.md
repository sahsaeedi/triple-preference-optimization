<p align="center">
    <img alt="TPO_LOGO.jpg" src="img/TPO_LOGO.jpg" width="1000" height="">
</p>

<div align="center">
    
# TPO: Triple Preference Optimization
</div>

<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="" alt="TPO paper"><img src="https://img.shields.io/badge/TPO-Paper-D9AB42" /></a>
<a href="https://www.asu.edu/" alt="jhu"><img src="https://img.shields.io/badge/Arizona_State_University-BEC23F" /></a>
<a href="https://twitter.com/sahsaeedi"><img src="https://img.shields.io/twitter/follow/sahsaeedi?style=social&logo=twitter" alt="follow on Twitter"></a>
<a href="https://twitter.com/ver_shivanshu"><img src="https://img.shields.io/twitter/follow/ver_shivanshu?style=social&logo=twitter" alt="follow on Twitter"></a>

 **TPO** (**T**riple **P**reference **O**ptimization) is a new preference learning method designed to align an LLM with three preferences without requiring a separate supervised fine-tuning step. The simple one-step combination of SFT and Preference Optimization outperforms current state-of-the-art alignment methods such as DPO, CPO, KTO, IPO and ORPO.



<p align="center">
    <img alt="TPO" src="img/demonstrations2.png" width="1000" height="">
</p>

## Contents 📄

- [Environment Setup](#environment-setup-)
- [Evaluation](#training-with-tpo-)
- [Training](#training-with-tpo-)
- [Data Information](#training-with-tpo-)

## Supports 🌟
- **Models**: Compatible with various models including but not limited to `alignment-handbook/zephyr-7b-sft-full` and `mistralai/Mistral-7B-v0.1`.
- **GPUs**: Optimized for both Nvidia GPUs.
- **Batch Sizes**: Configurable batch sizes for training to optimize GPU usage.
- **Data Parallelism**: Support for data parallelism to speed up training on multiple GPUs.

## To-Do List ✅
- [x] TPO script supports LoRA 
- [x] TPO script supports deepspeed 
- [x] TPO script supports accelerate
- [x] TPO script supports flash attention
- [ ] TPO script supports FSDP


## Environment Setup 🔧
This is a quick tutorial to set up and train a model with the TPO method.

**Create and activate a Conda environment**:
```bash
conda create --prefix tpo python=3.10 
conda activate tpo
```
**Install PyTorch with CUDA support (for Nvidia GPUs)**:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
**Install additional requirements**:
```bash
pip install -r requirements.txt
```

## Evaluation 🔁

We evaluated the models in both
single-turn and multi-turn scenarios using the <a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md">MT-Bench</a> benchmark.

We also assessed all alignment methods using
Open LLM Leaderboard benchmarks (
**ARC, HellaSwag, MMLU, TruthfulQA,** and **Winogrande**), 
 **Big Bench** benchmarks (
Causal Judgment (causal reasoning), Sports Understanding (commonsense reasoning), Formal Fallacies) and OpenBookQA using <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463">this version</a> of the Eleuther AI Harness.
## Training with TPO 🔥
Run  `run_tpo.sh` to train a model with TPO. This study used `alignment-handbook/zephyr-7b-sft-full`, `mistralai/Mistral-7B-v0.1` and `microsoft/phi-2` models. However, you can use other models to train with TPO.
```bash
#!/bin/bash

OUTPUT_DIR="/OUTPUT/DIR/PATH"

accelerate launch \
 --config_file configs/deepspeed_train_config_bf16.yaml \
  run_tpo.py  \
    --model_name_or_path mistralai/Mistral-7B-v0.1   \
    --tokenizer_name mistralai/Mistral-7B-v0.1   \
    --beta 0.2 \
    --alpha 0.9  \
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

For training other alignment methods, we used huggingface <a href="https://github.com/huggingface/alignment-handbook">alignment-handbook</a> repository.

## Dataset Information  🚀

To train
TPO, which requires three preferences, we created
a custom dataset from the <a href="https://huggingface.co/datasets/openbmb/UltraFeedback">**UltraFeedback**</a> dataset.
Here, the response with the highest score serves as
the reference response, the second-highest score as
the chosen response, and the lowest score as the
rejected response.

Finally, the dataset includes < $y_{ref}$, $y_w$, $y_l$ > where $y_{ref}$ represents reference response, $y_{w}$ represents chosen response and $y_l$ represents rejected response. 

The Data Format in JSON file must be:
```
{
    "prompt": PROMPT_SENTENCE,
    "reference": REFERENCE_SENTENCE,
    "chosen": CHOSEN_SENTENCE,
    "rejected": REJECTED_SENTENCE,
}
``` 

## BibTeX 📖

For more insights about various alignment methods, please check <a href="https://arxiv.org/abs/2404.14723"> paper</a>.
```
@misc{saeidi2024insights,
      title={Insights into Alignment: Evaluating DPO and its Variants Across Multiple Tasks}, 
      author={Amir Saeidi and Shivanshu Verma and Chitta Baral},
      year={2024},
      eprint={2404.14723},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
    
## Acknowledgement 🙏

- Thanks to <a href="https://github.com/huggingface"> Hugging Face</a>
 for their <a href="http://hf.co/docs/trl"> Transformer Reinforcement Learning (TRL) </a> library, which greatly assisted in our project. 
