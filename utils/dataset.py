from datasets import load_dataset
from datasets import load_from_disk
import numpy as np
import pandas as pd
import json

ds = load_dataset("openbmb/UltraFeedback")
def create_dataset(sample):
  try:
    socres = [sample['completions'][i]['overall_score'] for i in range(len(sample['models']))]
    ref_index = np.array(socres).argmax()
    chosen_index = np.argsort(socres, axis=0)[-2]
    rejected_index = np.array(socres).argmin()
    data = {
        'prompt': sample['instruction'],
        'reference': sample['completions'][ref_index]['response'],
        'chosen': sample['completions'][chosen_index]['response'],
        'rejected': sample['completions'][rejected_index]['response'],
        }
    return data
  except:
    data = {
        'prompt': None,
        'reference': None,
        'chosen': None,
        'rejected': None,
        }
    return data 

column_names = list(ds['train'].features)
dataset = ds['train'].map(
        create_dataset,
        num_proc=4,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

df = pd.DataFrame()
df['prompt'] = dataset['prompt']
df['reference'] = dataset['reference']
df['chosen'] = dataset['chosen']
df['rejected'] = dataset['rejected']
df = df.dropna()

df = df.sample(n=12000)
data = {
        'prompt': list(df['prompt'].values),
        'reference': list(df['reference'].values),
        'chosen': list(df['chosen'].values),
        'rejected': list(df['rejected'].values),
        }


with open("/TO/SAVE/DATASET/UltraFeedback_triple_preferences.json", "w") as outfile: 
    json.dump(data, outfile)

