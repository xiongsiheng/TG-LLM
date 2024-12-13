import sys
import json
import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset, load_dataset, concatenate_datasets
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse
from trl import DataCollatorForCompletionOnlyLM
import wandb


# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 
model_path = "meta-llama/Llama-2-13b-hf"



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train', action='store_true', help="Flag to enable training mode.")
    parser.add_argument('--test', action='store_true', help="Flag to enable testing mode.")
    parser.add_argument('--CoT_bs', action='store_true', help="Use CoT bootstrapping.")
    parser.add_argument('--data_aug', action='store_true', help="Use data augmentation.")
    parser.add_argument('--ICL', action='store_true', help="Flag to enable in-context learning during testing.")
    parser.add_argument('--overwrite', action='store_true', help="Flag to overwrite existing results.")
    parser.add_argument('--print_prompt', action='store_true', help="Flag to print example prompts.")
    parser.add_argument('--unit_test', action='store_true', help="Flag to enable unit testing with small data subsets.")
    parser.add_argument('--transferred', action='store_true', help="Use transfer learning results.")
    parser.add_argument('--no_TG', action='store_true', help="Disable temporal graphs as context.")
    parser.add_argument('--resume_from', type=str, help="Path to resume training from a checkpoint.")
    parser.add_argument('--prompt_format', type=str, default='plain', help="Format of the prompts: 'plain' or 'json'.")
    parser.add_argument('--use_wandb', action='store_true', help="Flag to enable logging with Weights & Biases.")
    return parser.parse_args()


def setup_wandb(args):
    """
    Sets up Weights and Biases (wandb) for tracking training and evaluation.
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    if args.use_wandb:
        wandb.init(
            project=f"{args.dataset}_TGR"
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def read_data(dataset_name, prefix, split, f_CoT_bs=False, f_data_aug=False):
    '''
    Read the data from the given file.

    args:
        dataset_name: string, dataset name
        filename: string, file name
        f_CoT_bs: bool, whether to use CoT bootstrapping
        f_data_aug: bool, whether to use data augmentation

    return:
        dataset: Dataset, the dataset
    '''
    file_path = f'../results/{dataset_name}_TGR_CoT_bs'
    if (not f_CoT_bs) or (not os.path.exists(file_path)):
        dataset = load_dataset(dataset_path, f'{dataset_name}_TGR')
        dataset = dataset[prefix + split]
    else:
        data = []
        for filename in os.listdir(f'{file_path}'):
            if not filename.startswith(f'{prefix + split}'):
                continue
            with open(f'{file_path}/{filename}') as json_file:
                data.append(json.load(json_file))

        # Convert list of dictionaries to the desired format
        data_dict = {'story': [item["story"] for item in data],
                     'TG': [item["TG"] for item in data],
                     'question': [item["question"] for item in data], 
                     'answer': [item["answer"] for item in data],
                     'external knowledge': [item["external knowledge"] for item in data],
                     'CoT': [CoT_sampling(item["CoT"], item['CoT_sample_prob']) for item in data],
                     'candidates': [item["candidates"] for item in data],
                     'id': [item['id'] for item in data],
                     'Q-Type': [item['Q-Type'] for item in data]}

        # Convert your data into a dataset
        dataset = Dataset.from_dict(data_dict)


    if f_data_aug and dataset_name in ['TGQA']:
        rel_entity_dict = collect_entity(dataset_name)
        random_entity_names = collect_entity_v2(dataset_name, rel_entity_dict)

        global_ent_mapping = {}   # we use a global mapping to ensure the consistency of entities and avoid confusion
        global_names_cnt = {}
        global_time_offset = random.randint(-20, 5)

        extra_data = [rel_entity_dict, global_ent_mapping, global_names_cnt, random_entity_names, global_time_offset]

        data_aug_dict = {'story': [], 'TG': [], 'external knowledge': [], 'question': [], 'CoT': [], 'candidates': [], 'answer': [], 'id': [], 'Q-Type': []}
        for sample in dataset:
            TG, EK, Q, CoT, C, A = data_augmentation(dataset_name, sample['TG'], sample['external knowledge'], sample['question'], sample['CoT'], 
                                                    sample['candidates'], sample['answer'], 
                                                    flag_rm_irr_edges=True, flag_change_relations=True, 
                                                    flag_change_entities=True, flag_change_times=True, extra_data=extra_data)
            data_aug_dict['story'].append(sample['story'])
            data_aug_dict['TG'].append(TG)
            data_aug_dict['external knowledge'].append(EK)
            data_aug_dict['question'].append(Q)
            data_aug_dict['CoT'].append(CoT)
            data_aug_dict['candidates'].append(C)
            data_aug_dict['answer'].append(A)
            data_aug_dict['id'].append(sample['id'])
            data_aug_dict['Q-Type'].append(sample['Q-Type'])

        dataset_aug = Dataset.from_dict(data_aug_dict)
        dataset = concatenate_datasets([dataset, dataset_aug])

    return dataset


def load_data(args):
    """
    Load and prepare datasets.
    
    args:
        args: argparse.ArgumentParser, command-line arguments

    return:
        data_train: Dataset, training dataset
        data_val: Dataset, validation dataset
        data_test: Dataset, testing dataset
        dataset_name: string, dataset name
        split_name: string, split name
    """
    dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
    split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
    prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]

    data_train = read_data(dataset_name, prefix, 'train', args.CoT_bs, args.data_aug)
    data_val = read_data(dataset_name, prefix, 'val', args.CoT_bs, args.data_aug)
    data_test = read_data(dataset_name, prefix, 'test')

    return data_train, data_val, data_test, dataset_name, split_name


def prepare_TG_pred(args, dataset_name):
    '''
    Prepare the predicted temporal graphs for testing.
    '''
    if not args.no_TG:
        # use estimated temporal graph for test
        path_TG_pred = f'../results/{dataset_name}_story_TG_trans/'
        if args.transferred:
            path_TG_pred = f'../results/TGQA_to_{dataset_name}_story_TG_trans/'
        pred_TGs = obtain_TG_pred(path_TG_pred, prompt_format=args.prompt_format)
    else:
        pred_TGs = {}
    return pred_TGs


def prepare_data(dataset_name, split_name, data_train, data_val, args):
    """
    Add prompts to datasets.
    
    args:
        dataset_name: string, dataset name
        split_name: string, split name
        data_train: Dataset, training dataset
        data_val: Dataset, validation dataset
        args: argparse.ArgumentParser, command-line arguments

    return:
        data_train: Dataset, training dataset with prompts
        data_val: Dataset, validation dataset with prompts
    """
    def add_prompt(sample):
        sample['prompt'] = my_generate_prompt_TG_Reasoning(dataset_name, split_name, sample['story'], sample['TG'], 
                                                           sample['external knowledge'], sample['question'], 
                                                           sample['CoT'], sample['answer'], args.ICL, 
                                                           eos_token="</s>", f_no_TG=args.no_TG, 
                                                           prompt_format=args.prompt_format)
        return sample

    data_train = data_train.map(add_prompt)
    data_val = data_val.map(add_prompt)
    return data_train, data_val


def print_example_prompts(data, pred_TGs, mode, dataset_name, split_name, args):
    ''''
    Print example prompts for debugging or inspection.

    args:
        data: Dataset, dataset
        pred_TGs: dict, predicted temporal graphs
        mode: string, mode of operation ('train' or 'test')
        dataset_name: string, dataset name
        split_name: string, split name
        args: argparse.ArgumentParser, command

    return:
        None
    '''
    for i in range(5):
        sample = data[i]
        if mode == 'train':
            prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, sample['story'], sample['TG'], 
                                                    sample['external knowledge'], 
                                                    sample['question'], sample['CoT'], sample['answer'], 
                                                    args.ICL, mode='train', eos_token="</s>", f_no_TG=args.no_TG, 
                                                    prompt_format=args.prompt_format)
        else:
            story_id = process_id(dataset_name, sample['id'])
            pred_TG = pred_TGs[story_id] if story_id in pred_TGs else None
            if pred_TG is None and not args.no_TG:
                continue
            prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, sample['story'], pred_TG, sample['external knowledge'], 
                                                        sample['question'], None, None, args.ICL, 
                                                        Q_type=sample['Q-Type'], mode='test', f_no_TG=args.no_TG, 
                                                        prompt_format=args.prompt_format)
        
        print(prompt)
        print('===============================')


def load_model_and_tokenizer():
    '''
    Load the model and tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def train_model(model, tokenizer, data_train, data_val, args, output_dir, max_steps=-1):
    """
    Train the model with SFT and LoRA.
    
    args:
        model: AutoModelForCausalLM, model
        tokenizer: AutoTokenizer, tokenizer
        data_train: Dataset, training dataset
        data_val: Dataset, validation dataset
        args: argparse.ArgumentParser, command-line arguments
        output_dir: string, output directory
        max_steps: int, maximum number of training steps

    return:
        None
    """
    def formatting_func(sample):
        return [p for p in sample['prompt']]
    
    response_template = "### Output"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    batch_size = 24 if args.dataset == 'TimeQA' else 12
    max_seq_length = 700 if args.dataset == 'TimeQA' else 1500

    SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, 
                  max_seq_length, max_steps=5 if args.unit_test else max_steps, 
                  resume_from_checkpoint=args.resume_from, collator=collator)


def test_model(model, tokenizer, data_test, dataset_name, strategy, split_name, pred_TGs, args, batch_size):
    """
    Run model inference and save results.
    
    args:
        model: AutoModelForCausalLM, model
        tokenizer: AutoTokenizer, tokenizer
        data_test: Dataset, testing dataset
        dataset_name: string, dataset name
        strategy: string, strategy
        split_name: string, split name
        pred_TGs: dict, predicted temporal graphs
        args: argparse.ArgumentParser, command-line arguments,
        batch_size: int, batch size
        
    return:
        None
    """
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    peft_model_id = f"../model_weights/{dataset_name}_{strategy}{split_name}/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)
    peft_model.eval()

    folder_path = f'../results/{dataset_name}_{strategy}{split_name}'
    os.makedirs(folder_path, exist_ok=True)

    if args.prompt_format.lower() == 'json':
        identifier='{\n"Thought":'
    else:
        identifier='Thought:'

    input_prompts, file_paths, samples = [], [], []
    for i in tqdm(range(len(data_test))):
        file_path = f"{folder_path}/{i}.json"
        if os.path.exists(file_path) and not args.overwrite:
            continue

        sample = data_test[i]
        story_id = process_id(dataset_name, sample['id'])
        pred_TG = pred_TGs[story_id] if story_id in pred_TGs else None
        cur_prompt = my_generate_prompt_TG_Reasoning(dataset_name, split_name, sample['story'], pred_TG, 
                                                     sample['external knowledge'], sample['question'], None, 
                                                     None, args.ICL, Q_type=sample['Q-Type'], mode='test', 
                                                     f_no_TG=args.no_TG, prompt_format=args.prompt_format)

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, prompt_format=args.prompt_format, identifier=identifier)
            input_prompts, file_paths, samples = [], [], []

    if input_prompts:
        run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, prompt_format=args.prompt_format, identifier=identifier)


def main():
    args = parse_arguments()
    data_train, data_val, data_test, dataset_name, split_name = load_data(args)

    pred_TGs = prepare_TG_pred(args, dataset_name)

    if args.print_prompt:
        print_example_prompts(data_train if args.train else data_test, pred_TGs, 'train' if args.train else 'test', dataset_name, split_name, args)
            
    setup_wandb(args)

    # Load tokenizer and model
    model, tokenizer = load_model_and_tokenizer()

    strategy = 'TGR' if not args.no_TG else 'storyR'
    output_dir = f"../model_weights/{dataset_name}_{strategy}{split_name}"

    if args.train:
        data_train, data_val = prepare_data(dataset_name, split_name, data_train, data_val, args)
        train_model(model, tokenizer, data_train, data_val, args, output_dir, max_steps=-1)    # early stop by setting max_steps 

    if args.test:
        test_model(model, tokenizer, data_test, dataset_name, strategy, split_name, pred_TGs, args, batch_size=24)



if __name__ == "__main__":
    main()