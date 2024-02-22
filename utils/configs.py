import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adatpers.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    multi_gpu_one_model: bool = field(
        default=False,
        metadata={
            "help": "Use multiple GPUs to load one model."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )


@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class TPOConfig(transformers.TrainingArguments):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    alpha: Optional[float] = field(
        default=1,
        metadata={"help": "The alpha factor in TPO loss. Higher alpha means increase effect of gold response learning in policy model."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})