# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List, Literal, Optional
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
import json


def get_datasets() -> Dataset:

    # ds = load_dataset("openbmb/UltraFeedback")
    # column_names = list(ds['train'].features)

    # dataset = ds['train'].map(
    #         prepare_dataset,
    #         num_proc=4,
    #         remove_columns=column_names,
    #         desc="Formatting comparisons with prompt template",
    #     )
    
    # dataset.save_to_disk("/data/data/amir/test/tpo_dataset_test.hf")
    # raw_datasets = load_from_disk("/data/data/amir/test/tpo_dataset_test.hf")
    # print(raw_datasets['prompt'][0])
    # print(raw_datasets['chosen'][0])
    # print(raw_datasets['rejected'][0])
    # df = pd.DataFrame()
    # df['prompt'] = dataset['prompt']
    # df['reference'] = dataset['reference']
    # df['chosen'] = dataset['chosen']
    # df['rejected'] = dataset['rejected']
    # print(df['prompt'][0])
    # print(dataset['prompt'][10])

    # dataset.save_to_disk("/home/ssaeidi1/triple_preferences_optimization/tpo_dataset_2.hf")
    # raw_datasets = load_from_disk('/home/ssaeidi1/triple_preferences_optimization/tpo_dataset_2.hf')
    with open("/home/ssaeidi1/triple_preferences_optimization/data/UltraFeedback_triple_preferences.json") as infile:
        dataset = json.load(infile)
    # print(dataset['prompt'][0])
    dataset = Dataset.from_dict(dataset)
    print(dataset)
    return dataset

if __name__ == "__main__":
    # raw_datasets = load_from_disk('/data/data/amir/tpo_dataset/tpo_dataset_2.hf')[:12000]
    # print(raw_datasets['prompt'][0])
    # print(raw_datasets['chosen'][0])
    # print(raw_datasets['rejected'][0])
    get_datasets()
