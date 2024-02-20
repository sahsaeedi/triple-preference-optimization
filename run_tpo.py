import logging
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import torch
import json
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import os
from utils.tpo_trainer import TPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from datasets import load_from_disk


def main():

    raw_datasets = load_from_disk('/data/data/amir/tpo_dataset/tpo_dataset_2.hf')[:12000]
    raw_datasets = Dataset.from_dict(raw_datasets)
    column_names = list(raw_datasets.features)

    # path =  "alignment-handbook/zephyr-7b-sft-full"
    path =  "mistralai/Mistral-7B-v0.1"

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map="auto")
    # model_ref = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    # model_ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map="auto")
    # model_ref = model
    tokenizer = AutoTokenizer.from_pretrained(path, device_map="auto")
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    tokenizer.truncation_side = 'left'

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "tpo"},
        num_proc=4,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    
    raw_datasets = raw_datasets.train_test_split(test_size=0.10)

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected", "text_reference": "reference"}
        )


    output_dir = '/data/data/amir/tpo/tpo_zephyr_01_1_mistral'
    # output_dir = "/data/data/amir/dpo/gpt_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.config.use_cache = False

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # max_steps=1000,
        logging_steps=10,
        save_steps=100,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5.0e-7,
        evaluation_strategy="steps",
        eval_steps=100,
        output_dir=output_dir,
        lr_scheduler_type="cosine",
        # warmup_steps=100,
        optim="adamw_torch",
        # optim="paged_adamw_32bit",
        # remove_unused_columns=False,
        run_name="tpo_zephyr_0.1_1_mistral",
        report_to = "wandb",
        bf16=True,
        log_level = 'info',
        num_train_epochs = 1,
        save_total_limit = 1,
        seed = 42,
        warmup_ratio = 0.1, 
    )

    tpo_trainer = TPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        alpha=0.5,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['test'],
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_length=1024,
    )

    # 6. train
    tpo_trainer.train()
    tpo_trainer.save_model(output_dir)

    # 7. save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    tpo_trainer.model.save_pretrained(output_dir)



if __name__ == "__main__":
    main()