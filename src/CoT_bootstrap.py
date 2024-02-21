import sys
import json
import random
import os
import re
import argparse
import numpy as np
import copy



import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from peft import PeftModel
from datasets import Dataset



os.environ["WANDB_DISABLED"] = "true"








def read_data(dataset_name, filename):
    file_path = f'../dataset/{dataset_name.split('_')[0]}/{filename}'
    with open(file_path) as json_file:
        data = json.load(json_file)

    # Convert list of dictionaries to the desired format
    data_dict = {'story': [item["story"] for item in data],
                 'TG': [item["TG"] for item in data],
                 'question': [item["question"] for item in data], 
                 'answer': [item["answer"] for item in data],
                 'EK': [item["EK"] if "EK" in item else None for item in data],
                 'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                 'candidates': [item["candidates"] if "candidates" in item else None for item in data],
                 'id': [item['id'] for item in data],
                 'Q-Type': [item['Q-Type'] if 'Q-Type' in item else None for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset




dataset_selection = 0
dataset_name = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'][dataset_selection]


filename_train = ['TGSR_train.json', 'TGSR_easy_train.json', 'TGSR_hard_train.json', 'TGSR_l2_train.json', 'TGSR_l3_train.json'][dataset_selection]
data_train = read_data(dataset_name, filename_train)

filename_val = ['TGSR_val.json', 'TGSR_easy_val.json', 'TGSR_hard_val.json', 'TGSR_l2_val.json', 'TGSR_l3_val.json'][dataset_selection]
data_val = read_data(dataset_name, filename_val)



print(data_train)
print(data_val)




def my_generate_prompt(TG, EK, Q):
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    prompt = f"Timeline:\n{TG}\n\nQuestion: {Q}"

    if EK is not None:
        if isinstance(EK, list):
            EK = '\n'.join(EK)
        prompt += f"\n\nUseful information:\n{EK}"

    prompt += "\n\nAnswer: Let's think step by step.\n\n"

    return prompt




for i in range(5):
    sample = data_train[i]
    prompt = my_generate_prompt(sample['TG'], sample['EK'], sample['question'])
    print(prompt)
    print('===============================')




model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )




def process_CoT(ans_pred):
    ans_pred = ans_pred.strip()
    for identifier in [' the answer is ', 'Answer:', ' answer is:', ' the correct answer is', ' the answers are ']:
        if identifier in ans_pred:
            ans_pred = ans_pred.split(identifier)[0].strip()
            break
    return ans_pred + ' the answer is '



def one_batch(input_prompts, samples):
    gamma = 0.5
    for j in range(len(input_prompts)):
        context_len = tokenizer(input_prompts[j], return_tensors="pt")["input_ids"].shape[1]
        cur_sample = samples[j]
        scores = []
        for CoT in cur_sample['CoT']:
            cur_prompt = input_prompts[j] + process_CoT(CoT)
            Probs_neg = []
            for cand in cur_sample['candidates']:
                input_tokens = tokenizer(cur_prompt + cand, return_tensors="pt")["input_ids"].to("cuda")
                target_ids = input_tokens.clone()
                target_ids[:, :context_len] = -100
                with torch.no_grad():
                    outputs = model(input_tokens, labels=target_ids)
                    loss = outputs.loss.cpu().numpy()
                    Probs_neg.append(loss)
            # print(Probs_neg)
            Probs_neg = np.mean(Probs_neg)


            input_tokens = tokenizer(cur_prompt + cur_sample['answer'], return_tensors="pt")["input_ids"].to("cuda")
            target_ids = input_tokens.clone()
            target_ids[:, :context_len] = -100
            with torch.no_grad():
                outputs = model(input_tokens, labels=target_ids)
                loss = outputs.loss.cpu().numpy()
                Probs_pos = copy.copy(loss)
                # print(Probs_pos)

            scores.append(Probs_pos + gamma*(Probs_pos - Probs_neg))

        scores = [np.exp(-10*s) for s in scores]
        cur_sample['CoT_sample_prob'] = (scores/np.sum(scores)).tolist()


    return samples




def CoT_bootstrap(data, filename):
    folder_path = f'../results/{dataset_name}_SR_bs'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    data_new = []
    input_prompts = []
    input_samples = []

    for i in range(len(data)):
        sample = data[i]
        cur_prompt = my_generate_prompt(sample['TG'], sample['EK'], sample['question'])
        input_prompts.append(cur_prompt)
        input_samples.append(sample)

        if len(input_prompts) >= 4:
            samples = one_batch(input_prompts, input_samples)
            data_new += samples
            input_prompts = []
            input_samples = []


    if len(input_prompts) > 0:
        samples = one_batch(input_prompts, input_samples)
        data_new += samples


    file_path = f'{folder_path}/{filename}'
    with open(file_path, 'w') as json_file:
        json.dump(data_new, json_file)


CoT_bootstrap(data_train, filename_train)
CoT_bootstrap(data_val, filename_val)