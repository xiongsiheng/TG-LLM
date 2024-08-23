import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from tqdm import tqdm
from utlis import *
from Models import *
from prompt_generation import *
import argparse

os.environ["WANDB_DISABLED"] = "true"




parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--ICL', action='store_true')
parser.add_argument('--shorten_story', action='store_true')
parser.add_argument('--hard_mode', action='store_true')
parser.add_argument('--print_prompt', action='store_true')
parser.add_argument('--unit_test', action='store_true')
parser.add_argument('--transferred_dataset', type=str)
parser.add_argument('--transferred', action='store_true')
parser.add_argument('--resume_from', type=str)
parser.add_argument('--prompt_format', type=str, default='plain')

args = parser.parse_args()



######### Config #########

dataset_name = args.dataset   # 'TGQA', 'TimeQA', 'TempReason'

f_train = args.train   # whether train the model
resume_from_checkpoint = args.resume_from  # set this to the checkpoint path if you want to resume training from a checkpoint otherwise leave it as None

f_test = args.test  # whether test the model
f_ICL = args.ICL  # whether use in-context learning during test
f_overwrite = args.overwrite  # whether overwrite existing test results

prompt_format = args.prompt_format  # whether use plain (text) or json as prompt format
f_shorten_story = args.shorten_story   # whether shorten the story (For TimeQA and TempReason, it is possible that the story is too long to feed into the model)
f_hard_mode = args.hard_mode   # whether use hard mode (only know relations) v.s. easy mode (know entities, relations and times) for translation

# If we want to test the transfer learning performance, just change the transferred dataset name.
# Note: current dataset_name should be 'TGQA', transferred_dataset_name = None (no transfer learning) or 'TimeQA' or 'TempReason'
transferred_dataset_name = args.transferred_dataset
f_transferred = args.transferred  # whether to use transfer learning model during test (if True, we will read the model weights learned from the transferred dataset)

f_print_example_prompt = args.print_prompt  # whether to print the example prompt for the model
f_unit_test = args.unit_test   # whether to run the unit test (only for debugging)

###########################





dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_Story_TG_Trans')


data_train = dataset['train']
data_val = dataset['val']
data_test = dataset['test']




def add_prompt(sample):
    sample['prompt'] = my_generate_prompt_TG_trans(dataset_name, sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times'], 
                                                   f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, max_story_len=1200, prompt_format=prompt_format)
    return sample

data_train = data_train.map(add_prompt)
data_val = data_val.map(add_prompt)


if f_unit_test:
    data_train = create_subset(data_train, 10)
    data_val = create_subset(data_val, 10)
    data_test = create_subset(data_test, 10)


print(data_train)
print(data_val)
print(data_test)



if f_print_example_prompt:
    for i in range(5):
        if f_train:
            sample = data_train[i]
            prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='train', eos_token="</s>", prompt_format=prompt_format)
        if f_test:
            sample = data_test[i]
            prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], None, sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='test', eos_token="", prompt_format=prompt_format)
        print(prompt)
        print('===============================')




model_name = "meta-llama/Llama-2-13b-hf"  # can be changed to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)



model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )

# this should be set for finutning and batched inference
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))


if f_train:
    def formatting_func(sample):
        '''Given a sample, obtain the prompt for the model'''
        output = []
        for p in sample['prompt']:
            output.append(p)
        return output

    output_dir = f"../model_weights/{dataset_name}_story_TG_trans"
    if transferred_dataset_name is not None:
        output_dir = f"../model_weights/{dataset_name}_to_{transferred_dataset_name}_story_TG_trans"

    max_steps = 5 if f_unit_test else -1
    response_template = "### Output"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  # By using this collator, we finetune the model on the output part only.
    batch_size = 12 if dataset_name == 'TGQA' else 4
    max_seq_length = 1500 if dataset_name == 'TGQA' else 4096
    SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length, max_steps, resume_from_checkpoint=resume_from_checkpoint, 
                  collator=collator)



if f_test:
    # we need padding on the left side to create the embeddings for a whole batch
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_story_TG_trans/final"
    if f_transferred:
        peft_model_id = f"../model_weights/TGQA_to_{dataset_name}_story_TG_trans/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")
    peft_model.eval()  # Set the model to evaluation mode

    folder_path = f'../results/{dataset_name}_story_TG_trans'
    if f_transferred:
        folder_path = f'../results/TGQA_to_{dataset_name}_story_TG_trans'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 4
    max_new_tokens = 1024 if dataset_name in 'TGQA' else 512  # Depends on the size of the (relevant) temporal graph
    input_prompts, file_paths, samples = [], [], []
    for i in tqdm(range(len(data_test))):
        file_path = folder_path + f'/{str(i)}.json'
        if (os.path.exists(file_path)) and (not f_overwrite):
            continue

        sample = data_test[i]
        cur_prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], None, sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='test', eos_token='', prompt_format=prompt_format)
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)
 
        # collect the prompts as a batch
        if len(input_prompts) >= batch_size:
            run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens, prompt_format=prompt_format)
            input_prompts, file_paths, samples = [], [], []

    # Last batch that is less than batch_size
    if len(input_prompts) > 0:
        run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens, prompt_format=prompt_format)