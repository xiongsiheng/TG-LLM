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
from datasets import Dataset
import openai
import time


os.environ["WANDB_DISABLED"] = "true"






dataset_selection = 0
model_selection = 2
f_using_CoT = 0 # whether use CoT
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]



def read_data(dataset_name, filename):
    file_path = f'../dataset/{dataset_name}/{filename}'
    with open(file_path) as json_file:
        data = json.load(json_file)

    # Convert list of dictionaries to the desired format
    data_dict = {'story': [item["story"] for item in data],
                 'Q': [item["question"] for item in data], 
                 'A': [item["answer"] for item in data],
                 'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                 'C': [item["candidates"] if "candidates" in item else None for item in data],
                 'id': [item['id'] for item in data],
                 'Q-Type': [item['Q-Type'] if 'Q-Type' in item else None for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset



data_test = read_data(dataset_name, filename)
print(data_test)





def my_gpt_completion(openai_model, messages, timeout, max_tokens=128, wait_time=0):
    openai.api_key =   # add your own api_key
    completion = openai.ChatCompletion.create(model=openai_model,
                                              messages=messages,
                                              request_timeout = timeout,
                                              temperature=0.7,
                                              max_tokens=max_tokens
                                            )
    response = completion['choices'][0]["message"]["content"]
    time.sleep(wait_time)

    return response




def my_generate_prompt(story, Q, C, Q_type=None):
    if f_ICL:
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'
        if not f_using_CoT:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_SP_{Q_type}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_CoT_{Q_type}.txt'
        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    story = story.replace('\n', ' ')

    if '(' not in C[0] and ')' not in C[0]:
        C = ['( ' + cand + ' )' for cand in C]
    Q += ' Choose from ' + ', '.join(C) + '.'

    if f_ICL:
        prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}"
    else:
        prompt = f"Story: {story}\n\nQuestion: {Q}"


    if f_using_CoT:
        prompt += "\n\nAnswer: Let's think step by step.\n\n"
    else:
        prompt += "\n\nAnswer: "

    return prompt





for i in range(5):
    sample = data_test[i]
    prompt = my_generate_prompt(sample['story'], sample['Q'], sample['C'], Q_type=sample['Q-Type'])

    print(prompt)
    print('===============================')






if 'Llama' in model_name:
    model_name_cmp = f'meta-llama/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name_cmp,
                                                load_in_8bit=True,
                                                device_map="auto"
                                                )





def one_batch(input_prompts, samples, file_paths, max_new_tokens=512):
    if 'Llama' in model_name:
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
              )


        for j in range(len(input_prompts)):
            op = tokenizer.decode(generation_output[j], skip_special_tokens=True)
            op = op[len(input_prompts[j]):]
            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    if 'gpt' in model_name:
        for j in range(len(input_prompts)):
            messages = []
            messages.append({"role": "user", "content": input_prompts[j]})
            op = my_gpt_completion(model_name, messages, 600, max_tokens=max_new_tokens, wait_time=0.5)

            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    return







folder_path = f'../results/{dataset_name}_ICL_{model_name}'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


input_prompts = []
file_paths = []
samples = []
for i in range(len(data_test)):
    file_path = folder_path + f'/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt(sample['story'], sample['Q'], sample['C'], Q_type=sample['Q-Type'])

    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    if len(input_prompts) >= 8:
        one_batch(input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    one_batch(input_prompts, samples, file_paths)