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
    file_path = f'../dataset/{dataset_name}/{filename}'
    with open(file_path) as json_file:
        data = json.load(json_file)

    # Convert list of dictionaries to the desired format
    data_dict = {'story': [item["story"] for item in data],
                 'TG': [item["TG"] for item in data],
                 'Q': [item["question"] for item in data], 
                 'A': [item["answer"] for item in data],
                 'EK': [item["EK"] if "EK" in item else None for item in data],
                 'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                 'C': [item["candidates"] if "candidates" in item else None for item in data],
                 'id': [item['id'] for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset




dataset_selection = 0
f_train = 1
f_test = 1



dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]





filename = ['TGSR_train.json', 'TGSR_easy_train.json', 'TGSR_hard_train.json', 'TGSR_l2_train.json', 'TGSR_l3_train.json'][dataset_selection]
data_train = read_data(dataset_name, filename)

filename = ['TGSR_val.json', 'TGSR_easy_val.json', 'TGSR_hard_val.json', 'TGSR_l2_val.json', 'TGSR_l3_val.json'][dataset_selection]
data_val = read_data(dataset_name, filename)

filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
data_test = read_data(dataset_name, filename)


print(data_train)
print(data_val)
print(data_test)





if f_test:
    TG_pred = {}
    path_TG_pred = f'../results/{dataset_name}_story_TG_trans/'
    for filename in os.listdir(path_TG_pred):
        file_path = os.path.join(path_TG_pred, filename)
        with open(file_path) as json_file:
            data = json.load(json_file)
        TG_pred[data['id']] = data['prediction']


def process_id(sample_id):
    story_id = sample_id
    if dataset_name == 'TimeQA':
        story_id = story_id[:-2]
    if dataset_name == 'TimeQA':
        story_id = story_id[2:-2]
    return story_id


def my_generate_prompt(TG, EK, Q, CoT, A, eos_token="</s>"):
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    prompt = f"Timeline:\n{TG}\n\nQuestion: {Q}"

    if EK is not None:
        if isinstance(EK, list):
            EK = '\n'.join(EK)
        prompt += f"\n\nUseful information:\n{EK}"

    prompt += "\n\nAnswer: Let's think step by step.\n\n"

    if CoT is not None:
        if isinstance(CoT, list):
            CoT = CoT[0]
        prompt += CoT

    prompt += eos_token
    return prompt




for i in range(5):
    if f_train:
        sample = data_train[i]
        prompt = my_generate_prompt(sample['TG'], sample['EK'], sample['Q'], sample['CoT'], sample['A'])
    if f_test:
        sample = data_test[i]
        story_id = process_id(sample['id'])
        prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], sample['CoT'], sample['A'], eos_token="")
    print(prompt)
    print('===============================')





model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )







if f_train:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # this should be set for finutning and batched inference
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    # Loading in 8 bit ..."
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    output_dir = f"../model_weights/{dataset_name}_TGSR"
    per_device_train_batch_size = 12
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 12
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    max_steps = 30
    warmup_ratio = 0.03
    evaluation_strategy="steps"
    lr_scheduler_type = "constant"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size
            )


    def formatting_func(sample):
        output = []
        for g, e, q, cot, a in zip(sample['TG'], sample['EK'], sample['Q'], sample['CoT'], sample['A']):
            op = my_generate_prompt(g, e, q, cot, a)
            output.append(op)

        return output


    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"{output_dir}/final")



if f_test:
    def one_batch(tokenizer, input_prompts, samples, file_paths, max_new_tokens=512):
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = peft_model.generate(
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

        return



    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_TGSR/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

    folder_path = f'../results/{dataset_name}_TGSR'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    input_prompts = []
    file_paths = []
    samples = []
    for i in range(len(data_test)):
        file_path = folder_path + f'/{str(i)}.json'
        if os.path.exists(file_path):
            continue

        sample = data_test[i]
        story_id = process_id(sample['id'])
        cur_prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], None, None, eos_token='')


        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= 8:
            one_batch(tokenizer, input_prompts, samples, file_paths)
            input_prompts = []
            file_paths = []
            samples_info = []


    if len(input_prompts) > 0:
        one_batch(tokenizer, input_prompts, samples, file_paths)