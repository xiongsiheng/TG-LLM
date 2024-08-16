import sys
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse

os.environ["WANDB_DISABLED"] = "true"



parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--CoT', action='store_true')
parser.add_argument('--ICL', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--shorten_story', action='store_true')
parser.add_argument('--print_prompt', action='store_true')
parser.add_argument('--unit_test', action='store_true')


args = parser.parse_args()



######### Config #########

dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
model_selection = ['Llama2-7b', 'Llama2-13b', 'Llama2-70b'].index(args.model)
f_using_CoT = args.CoT  # whether use CoT
f_ICL = args.ICL   # whether use in-context learning during test
f_overwrite = args.overwrite   # whether overwrite existing test results
f_shorten_story = args.shorten_story   # whether shorten the story (For TimeQA and TempReason, it is possible that the story is too long to feed into the model)
f_print_example_prompt = args.print_prompt  # whether to print the example prompt for the model
f_unit_test = args.unit_test   # whether to run the unit test (only for debugging)

###########################




dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]
model_name = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]
learning_setting = 'SP' if not f_using_CoT else 'CoT'




dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
data_test = dataset[prefix + 'test']

if f_unit_test:
    data_test = create_subset(data_test, 10)

print(data_test)




if f_print_example_prompt:
    for i in range(5):
        sample = data_test[i]
        prompt = my_generate_prompt_ICL(dataset_name, split_name, learning_setting, sample['story'], sample['question'], sample['candidates'], 
                                        f_ICL, f_shorten_story, f_using_CoT, Q_type=sample['Q-Type'])

        print(prompt)
        print('===============================')



model_name_cmp = f'meta-llama/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'
model = AutoModelForCausalLM.from_pretrained(model_name_cmp,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )
model.eval()





folder_path = f'../results/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}_ppl' if f_ICL else f'../results/{dataset_name}{split_name}_{model_name}_ppl'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


if f_using_CoT:
    folder_path_past_res = f'../results/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}' if f_ICL else f'../results/{dataset_name}{split_name}_{model_name}'
    if not os.path.exists(folder_path_past_res):
        print('Error! Please first generate the CoT results.')
        sys.exit()


batch_size = 4
input_prompts = []
file_paths = []
samples = []
for i in tqdm(range(len(data_test))):
    file_path = f'{folder_path}/{str(i)}.json'
    if os.path.exists(file_path) and (not f_overwrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt_ICL(dataset_name, split_name, learning_setting, sample['story'], sample['question'], sample['candidates'], 
                                        f_ICL, f_shorten_story, f_using_CoT, Q_type=sample['Q-Type'])

    if f_using_CoT:
        file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
        if not os.path.exists(file_path_past_res):
            continue

        with open(file_path_past_res) as json_file:
            past_res = json.load(json_file)

        cur_prompt += process_CoT(past_res['prediction'])

    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    # collect the prompts as a batch
    if len(input_prompts) >= batch_size:
        run_one_batch_ppl(model, tokenizer, input_prompts, samples, file_paths, using_json=False)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    run_one_batch_ppl(model, tokenizer, input_prompts, samples, file_paths, using_json=False)