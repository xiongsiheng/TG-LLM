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






dataset_selection = 0
f_train = 1
f_test = 1
f_CoT_bs = 1
f_data_aug = 1
f_ICL = 1  # whether use in-context learning during test
f_rewrite = 1 # whether rewrite existing test results


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]





def data_augmentation(TG, EK, Q, CoT, C, A, flag_rm_irr_edges=False, flag_change_relations=False, flag_change_entities=False, flag_change_times=False):
    flag1, flag2 = 0, 0
    if isinstance(TG, list):
        TG = '\n'.join(TG)
        flag1 = 1

    if EK is not None:
        if isinstance(EK, list):
            EK = '\n'.join(EK)
            flag2 = 1

    if flag_rm_irr_edges:
        TG = rm_irr_edges(TG, CoT, Q)

    if flag_change_relations:
        TG = change_rels(TG)

    if flag_change_entities:
        TG, EK, Q, CoT, C, A = change_entities(TG, [TG, EK, Q, CoT, C, A])

    if flag_change_times:
        TG, EK, Q, CoT, C, A = change_times(TG, [TG, EK, Q, CoT, C, A])

    if flag1:
        TG = TG.split('\n')
    if flag2:
        EK = EK.split('\n')

    return TG, EK, Q, CoT, C, A




def rm_irr_edges(TG, CoT, Q):
    facts_ls = TG.split('\n')
    rm_facts = [i for (i, fact) in enumerate(facts_ls) if (fact[:-len(fact.split(')')[-1])].strip() not in CoT) and (fact[:-len(fact.split(')')[-1])].strip() not in Q)]
    random.shuffle(rm_facts)
    rm_facts = rm_facts[:int(0.2*len(rm_facts))]
    facts_ls = [fact for (i, fact) in enumerate(facts_ls) if i not in rm_facts]
    TG = '\n'.join(facts_ls)
    return TG




def change_rels(TG):
    with open(f'../materials/{dataset_name}/rel_synonyms.txt', 'r') as file:
        contents = file.readlines()

    rel_synonyms = {}
    for line in contents:
        if len(line.strip()) > 0:
            rel = line.split('\t')[0].strip()
            rel_synonyms[rel] = line.split('\t')


    facts_ls = TG.split('\n')
    facts_new_ls = []
    for fact in facts_ls:
        fact_front = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
        fact_back = fact[-len(fact.split(')')[-1]):]
        for rel in rel_synonyms:
            if ' ' + rel in fact_front:
                sub = fact_front.split(rel)[0].strip()
                obj = fact_front.split(rel)[1].strip()
                fact_new = f'({sub} {random.choice(rel_synonyms[rel]).strip()} {obj}){fact_back}'
                facts_new_ls.append(fact_new)
                break

    TG = '\n'.join(facts_new_ls)
    return TG



def change_times(TG, prompts):
    facts_ls = TG.split('\n')
    mapping_time = {}

    for fact in facts_ls:
        time = fact.split(' at ')[-1]
        mapping_time[time] = str(int(time) + global_time_offset)
        mapping_time[time] = mapping_time[time][0] + '_' + mapping_time[time][1:]

    prompts_new = []
    for prompt in prompts:
        if isinstance(prompt, list):
            prompt_new = []
            for sub_prompt in prompt:
                for time in mapping_time:
                    sub_prompt = sub_prompt.replace(time, mapping_time[time])
                sub_prompt = sub_prompt.replace('_', '')
                prompt_new.append(sub_prompt)
            prompt = copy.copy(prompt_new)
        else:
            for time in mapping_time:
                prompt = prompt.replace(time, mapping_time[time])
            prompt = prompt.replace('_', '')
        prompts_new.append(prompt)
    return prompts_new



def collect_entity():
    with open(f'../materials/{dataset_name}/rel_dict.txt', 'r') as file:
        contents = file.readlines()

    rel_entity_dict = {}
    for line in contents:
        if len(line.strip()) > 0:
            rel = line.split('\t')[0].strip()
            rel_entity_dict[rel] = {'sub': [], 'obj': []}

    def process_ent(rel_entity_dict, data):
        for i in range(len(data)):
            sample = data[i]
            facts_ls = sample['TG'].split('\n')
            for fact in facts_ls:
                fact = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
                for rel in rel_entity_dict:
                    if ' ' + rel in fact:
                        rel_entity_dict[rel]['sub'].append(fact.split(rel)[0].strip())
                        rel_entity_dict[rel]['obj'].append(fact.split(rel)[1].strip())
                        break
        return rel_entity_dict



    filename = ['TGSR_train.json', 'TGSR_easy_train.json', 'TGSR_hard_train.json', 'TGSR_l2_train.json', 'TGSR_l3_train.json'][dataset_selection]
    data_train = read_data(dataset_name, filename)

    filename = ['TGSR_val.json', 'TGSR_easy_val.json', 'TGSR_hard_val.json', 'TGSR_l2_val.json', 'TGSR_l3_val.json'][dataset_selection]
    data_val = read_data(dataset_name, filename)

    filename = ['TGSR_test.json', 'TGSR_easy_test.json', 'TGSR_hard_test.json', 'TGSR_l2_test.json', 'TGSR_l3_test.json'][dataset_selection]
    data_test = read_data(dataset_name, filename)



    rel_entity_dict = process_ent(rel_entity_dict, data_train)
    rel_entity_dict = process_ent(rel_entity_dict, data_val)
    rel_entity_dict = process_ent(rel_entity_dict, data_test)

    for rel in rel_entity_dict:
        rel_entity_dict[rel]['sub'] = list(set(rel_entity_dict[rel]['sub']))
        rel_entity_dict[rel]['obj'] = list(set(rel_entity_dict[rel]['obj']))

    sub_total = []
    for rel in rel_entity_dict:
        sub_total += rel_entity_dict[rel]['sub']
    sub_total = list(set(sub_total))
    rel_entity_dict['sub_total'] = sub_total

    return rel_entity_dict



def collect_entity_v2(existing_names=None):
    path = f'../materials/{dataset_name}/random_names'
    with open(f'{path}/sub_total.txt') as file:
        context = file.readlines()

    sub_total = [line.strip() for line in context]
    sub_total = list(set(sub_total))

    ent_names = {}
    for file_path in os.listdir(path):
        if file_path != 'sub_total.txt':
            with open(f'{path}/{file_path}') as file:
                context = file.read()
            rel = file_path.split('.')[0]
            ent_names[rel] = {}
            ent_names[rel]['obj'] = context.strip().split('\n')
            ent_names[rel]['obj'] = list(set(ent_names[rel]['obj']))


    if existing_names is not None:
        sub_total = [name for name in sub_total if name not in existing_names['sub_total']]
        random.shuffle(sub_total)
        for rel in ent_names:
            ent_names[rel]['obj'] = [name for name in ent_names[rel]['obj'] if name not in existing_names[rel]['obj']]
            random.shuffle(ent_names[rel]['obj'])
        ent_names['sub_total'] = sub_total

    return ent_names



def change_entities(TG, prompts):
    facts_ls = TG.split('\n')
    for fact in facts_ls:
        fact = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
        for rel in rel_entity_dict:
            if ' ' + rel in fact:
                sub = fact.split(rel)[0].strip()
                obj = fact.split(rel)[1].strip()
                if sub not in global_ent_mapping:
                    if 'sub_total' not in global_names_cnt:
                        global_names_cnt['sub_total'] = 0
                    global_ent_mapping[sub] = random_entity_names['sub_total'][global_names_cnt['sub_total']]
                    global_names_cnt['sub_total'] += 1

                if obj not in global_ent_mapping:
                    if rel in ['was married to']:
                        # mapping[obj] = create_new_person()
                        if 'sub_total' not in global_names_cnt:
                            global_names_cnt['sub_total'] = 0
                        global_ent_mapping[obj] = random_entity_names['sub_total'][global_names_cnt['sub_total']]
                        global_names_cnt['sub_total'] += 1

                    else:
                        if rel not in global_names_cnt:
                            global_names_cnt[rel] = 0

                        valid_name = random_entity_names[rel]['obj'][global_names_cnt[rel]]
                        while valid_name in global_ent_mapping.values():
                            global_names_cnt[rel] += 1
                            valid_name = random_entity_names[rel]['obj'][global_names_cnt[rel]]

                        global_ent_mapping[obj] = copy.copy(valid_name)
                break

    prompts_new = []
    for prompt in prompts:
        if isinstance(prompt, list):
            prompt_new = []
            for sub_prompt in prompt:
                for entity in global_ent_mapping:
                    sub_prompt = sub_prompt.replace(entity, global_ent_mapping[entity])
                prompt_new.append(sub_prompt)
            prompt = copy.copy(prompt_new)
        else:
            for entity in global_ent_mapping:
                prompt = prompt.replace(entity, global_ent_mapping[entity])
        prompts_new.append(prompt)
    return prompts_new



def CoT_sampling(CoT, CoT_sample_prob):
    return random.choices(CoT, weights=CoT_sample_prob, k=1)




def read_data(dataset_name, filename, f_CoT_bs=0, f_data_aug=0):
    if f_CoT_bs:
        file_path = f'../results/{dataset_name}_SR_bs/{filename}'
    else:
        file_path = f'../dataset/{dataset_name}/{filename}'

    with open(file_path) as json_file:
        data = json.load(json_file)

    if f_data_aug:
        data_aug = []
        for sample in data:
            TG, EK, Q, CoT, C, A = data_augmentation(sample['TG'], sample['EK'], sample['question'], sample['CoT'], sample['candidates'], sample['answer'], 
                                                      flag_rm_irr_edges=True, flag_change_relations=True, 
                                                      flag_change_entities=True, flag_change_times=True)
            data_aug.append({'story': sample['story'], 'TG': TG, 'EK': EK, 'question': Q, 'CoT': CoT, 'candidates': C, 'answer': A, 'id': sample['id'], 
                             'CoT_sample_prob': sample['CoT_sample_prob'], 'Q-Type': sample['Q-Type']})
        data += data_aug


    if f_CoT_bs:
        # Convert list of dictionaries to the desired format
        data_dict = {'story': [item["story"] for item in data],
                     'TG': [item["TG"] for item in data],
                     'Q': [item["question"] for item in data], 
                     'A': [item["answer"] for item in data],
                     'EK': [item["EK"] if "EK" in item else None for item in data],
                     'CoT': [CoT_sampling(item["CoT"], item['CoT_sample_prob']) if "CoT" in item else None for item in data],
                     'C': [item["candidates"] if "candidates" in item else None for item in data],
                     'id': [item['id'] for item in data],
                     'Q-Type': [item['Q-Type'] if 'Q-Type' in item else None for item in data]}
    else:
        # Convert list of dictionaries to the desired format
        data_dict = {'story': [item["story"] for item in data],
                     'TG': [item["TG"] for item in data],
                     'Q': [item["question"] for item in data], 
                     'A': [item["answer"] for item in data],
                     'EK': [item["EK"] if "EK" in item else None for item in data],
                     'CoT': [item["CoT"] if "CoT" in item else None for item in data],
                     'C': [item["candidates"] if "candidates" in item else None for item in data],
                     'id': [item['id'] for item in data],
                     'Q-Type': [item['Q-Type'] if 'Q-Type' in item else None for item in data]}

    # Convert your data into a dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset









if f_data_aug:
    rel_entity_dict = collect_entity()
    random_entity_names = collect_entity_v2(rel_entity_dict)

    global_ent_mapping = {}
    global_names_cnt = {}
    global_time_offset = random.randint(-20, 5)



filename = ['TGSR_train.json', 'TGSR_easy_train.json', 'TGSR_hard_train.json', 'TGSR_l2_train.json', 'TGSR_l3_train.json'][dataset_selection]
data_train = read_data(dataset_name, filename, f_CoT_bs, f_data_aug)

filename = ['TGSR_val.json', 'TGSR_easy_val.json', 'TGSR_hard_val.json', 'TGSR_l2_val.json', 'TGSR_l3_val.json'][dataset_selection]
data_val = read_data(dataset_name, filename, f_CoT_bs, f_data_aug)

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


def my_generate_prompt(TG, EK, Q, CoT, A, Q_type=None, mode=None, eos_token="</s>"):
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    if f_ICL and mode == 'test':
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'
        file_path = f'../materials/{dataset_name}/prompt_examples_TGSR_{Q_type}.txt'
        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_ICL and mode == 'test':
        prompt = f"Example:\n\n{prompt_examples}\n\nTest:\n\nTimeline:\n{TG}\n\nQuestion: {Q}"
    else:
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
        prompt = my_generate_prompt(sample['TG'], sample['EK'], sample['Q'], sample['CoT'], sample['A'], mode='train')

    if f_test:
        sample = data_test[i]
        story_id = process_id(sample['id'])
        prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], sample['CoT'], sample['A'], Q_type=sample['Q-Type'], mode='test', eos_token="")

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
    max_steps = 50
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
        for g, e, q, cot, a in zip(sample['TG'], sample['EK'], sample['Q'], sample['CoT'], sample['A'], mode='train'):
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
        if os.path.exists(file_path) and (not f_rewrite):
            continue

        sample = data_test[i]
        story_id = process_id(sample['id'])
        cur_prompt = my_generate_prompt(TG_pred[story_id], sample['EK'], sample['Q'], None, None, Q_type=sample['Q-Type'], mode='test', eos_token="")


        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= 8:
            one_batch(tokenizer, input_prompts, samples, file_paths)
            input_prompts = []
            file_paths = []
            samples = []


    if len(input_prompts) > 0:
        one_batch(tokenizer, input_prompts, samples, file_paths)