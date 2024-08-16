import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from Models import *
from tqdm import tqdm
from prompt_generation import *
import argparse

os.environ["WANDB_DISABLED"] = "true"



parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--print_prompt', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--unit_test', action='store_true')


args = parser.parse_args()





######### Config #########

dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
f_print_example_prompt = args.print_prompt   # whether to print the example prompt for the model
f_overwrite = args.overwrite   # whether overwrite existing results
f_unit_test = args.unit_test   # whether to run the unit test (only for debugging)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]

dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')

split_train = ['train', 'easy_train', 'hard_train', 'l2_train', 'l3_train'][dataset_selection]
data_train = dataset[split_train]

split_val = ['val', 'easy_val', 'hard_val', 'l2_val', 'l3_val'][dataset_selection]
data_val = dataset[split_val]


if f_unit_test:
    data_train = create_subset(data_train, 10)
    data_val = create_subset(data_val, 10)


print(data_train)
print(data_val)



if f_print_example_prompt:
    for i in range(5):
        sample = data_train[i]
        prompt = my_generate_prompt_CoT_bs(sample['TG'], sample['external knowledge'], sample['question'])
        print(prompt)
        print('===============================')



model_name = "meta-llama/Llama-2-13b-hf" # you can change to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )

model.eval()  # Set the model to evaluation mode



def CoT_bootstrap(data, filename, model, tokenizer):
    '''
    Given a list of CoT for each sample that leads to the correct final answer, calculate the probability of each CoT for each sample.
    Todo: Here we start from the data with filtered CoTs. We can also start from the data with no CoTs, and we need to generate and filter the CoTs in this function.

    Args:
    data: the data with filtered CoTs
    filename: the filename to save the results

    Returns:
    None
    '''
    folder_path = f'../results/{dataset_name}_TGR_CoT_bs'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 4
    file_path = f'{folder_path}/{filename}'
    input_prompts = []
    input_samples = []
    for sample in tqdm(data):
        cur_id = sample['id']
        cur_file_path = f'{file_path}_{cur_id}.json'
        if os.path.exists(cur_file_path) and (not f_overwrite):
            continue

        cur_prompt = my_generate_prompt_CoT_bs(sample['TG'], sample['external knowledge'], sample['question'])
        input_prompts.append(cur_prompt)
        input_samples.append(sample)

        if len(input_prompts) >= batch_size:
            run_one_batch_CoT_bs(model, tokenizer, input_prompts, input_samples, file_path)
            input_prompts = []
            input_samples = []

    # Last batch that is less than batch_size
    if len(input_prompts) > 0:
        run_one_batch_CoT_bs(model, tokenizer, input_prompts, input_samples, file_path)

    return


CoT_bootstrap(data_train, split_train, model, tokenizer)