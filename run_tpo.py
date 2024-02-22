import logging
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import torch
import json
import numpy as np
import sys
import random
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import os
import transformers
from utils.tpo_trainer import TPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from datasets import load_from_disk
from utils.configs import DataArguments, ModelArguments, TPOConfig
from utils.model_utils import load_model, get_tokenizer
from utils.data import load_dataset, apply_chat_template, get_datasets

logger = logging.getLogger(__name__)

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)


    # Load dataset
    raw_datasets = get_datasets()
    # logger.info(
    #     f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    column_names = list(raw_datasets.features)


    # Load tokenizer
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    # path =  "alignment-handbook/zephyr-7b-sft-full"
    # path =  "mistralai/Mistral-7B-v0.1"

    # Apply chat template
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "tpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    
    raw_datasets = raw_datasets.train_test_split(test_size=0.10)

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected", "text_reference": "reference"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")


    model = load_model(data_args, model_args, training_args, tokenizer, logger)


    # output_dir = '/data/data/amir/tpo/tpo_zephyr_01_1_mistral'
    # output_dir = "/data/data/amir/dpo/gpt_test"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    model.config.use_cache = False


    tpo_trainer = TPOTrainer(
        model,
        args=training_args,
        beta=training_args.beta,
        alpha=training_args.alpha,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['test'],
        tokenizer=tokenizer,
        max_prompt_length=training_args.max_length,
        max_length=training_args.max_prompt_length,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        tpo_trainer.train(resume_from_checkpoint=checkpoint)

        tpo_trainer.save_state()
        if model_args.use_peft:
            if torch.distributed.get_rank() == 0:
                model.save_pretrained(training_args.output_dir) 
        else:
            tpo_trainer.save_model()  # Saves the tokenizer too for easy upload

    # # 6. train
    # tpo_trainer.train()
    # tpo_trainer.save_model(output_dir)

    # # 7. save
    # output_dir = os.path.join(output_dir, "final_checkpoint")
    # tpo_trainer.model.save_pretrained(output_dir)



if __name__ == "__main__":
    main()