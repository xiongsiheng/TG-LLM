import sys
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse


os.environ["WANDB_DISABLED"] = "true"

# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 
model_path = "meta-llama/Llama-2-13b-hf"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, 
                        help="Dataset name to use. Options: TGQA, TimeQA_easy, TimeQA_hard, TempReason_l2, TempReason_l3")
    parser.add_argument('--ICL', action='store_true', help="Use in-context learning during testing")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing test results")
    parser.add_argument('--print_prompt', action='store_true', help="Print example prompts for debugging")
    parser.add_argument('--unit_test', action='store_true', help="Run unit tests with a subset of data")
    parser.add_argument('--transferred', action='store_true', help="Use transfer learning results")
    parser.add_argument('--no_TG', action='store_true', help="Disable temporal graphs as context")
    parser.add_argument('--prompt_format', type=str, default='plain', choices=['plain', 'json'],
                        help="Format of the prompts: 'plain' or 'json'")

    return parser.parse_args()


def setup_directories(dataset_name, strategy, split_name):
    """
    Creates result directories if they do not exist.

    Args:
        dataset_name (str): Name of the dataset.
        strategy (str): Strategy for reasoning.
        split_name (str): Name of the split.

    Returns:
        folder_path (str): Path to the results folder.
        folder_path_past_res (str): Path to the past results folder.
    """
    folder_path = f'../results/{dataset_name}_{strategy}{split_name}_ppl'
    os.makedirs(folder_path, exist_ok=True)

    folder_path_past_res = f'../results/{dataset_name}_{strategy}{split_name}'
    if not os.path.exists(folder_path_past_res):
        print('Error! Please first generate the CoT results.')
        sys.exit()
    return folder_path, folder_path_past_res


def print_prompt_sample(sample, pred_TGs, dataset_name, args):
    """
    Prints example prompts for debugging.

    Args:
        sample (dict): A single data sample containing story, question, etc.
        pred_TGs (dict): Predicted temporal graphs.
        dataset_name (str): Name of the dataset.
        args: Parsed command-line arguments.
    """
    story_id = process_id(dataset_name, sample['id'])
    pred_TG = pred_TGs.get(story_id, None)

    if pred_TG is None and not args.no_TG:
        return

    prompt = my_generate_prompt_TG_Reasoning(
        dataset_name, '', sample['story'], pred_TG,
        sample['external knowledge'], sample['question'],
        None, None, args.ICL, Q_type=sample['Q-Type'],
        mode='test', f_no_TG=args.no_TG, prompt_format=args.prompt_format
    )

    print(f"Sample ID: {story_id}")
    print("Generated Prompt:")
    print(prompt)
    print("=" * 30)


def load_model_and_tokenizer(model_name, peft_model_id):
    """
    Loads the base model, tokenizer, and LoRA model weights.

    Args:
        model_name (str): Name of the base model.
        peft_model_id (str): Path to the LoRA model weights.

    Returns:
        peft_model (PeftModel): Loaded LoRA model.
        tokenizer (AutoTokenizer): Loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    peft_model = PeftModel.from_pretrained(model, peft_model_id, 
                                           torch_dtype=torch.float16, 
                                           offload_folder="lora_results/lora_7/temp")
    peft_model.eval()
    return peft_model, tokenizer


def prepare_data(args):
    """
    Prepares the dataset and temporal graphs based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        data_test (Dataset): Test dataset.
        pred_TGs (dict): Predicted temporal graphs.
        dataset_name (str): Name of the dataset.
        split_name (str): Name of the split.
        prefix (str): Prefix for the split.
    """
    dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)

    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
    split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
    prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]

    dataset = load_dataset(dataset_path, f'{dataset_name}_TGR')
    data_test = dataset[f'{prefix}test']

    if args.unit_test:
        data_test = create_subset(data_test, 100)

    pred_TGs = {}
    if not args.no_TG:
        path_TG_pred = f'../results/{dataset_name}_story_TG_trans/'
        if args.transferred:
            path_TG_pred = f'../results/TGQA_to_{dataset_name}_story_TG_trans/'
        pred_TGs = obtain_TG_pred(path_TG_pred)

    return data_test, pred_TGs, dataset_name, split_name


def evaluate(peft_model, tokenizer, data_test, pred_TGs, args, folder_path, folder_path_past_res):
    """
    Generates prompts, evaluates model responses, and saves results in batches.
    """
    batch_size = 4
    input_prompts, file_paths, samples = [], [], []
    using_json = args.prompt_format.lower() == 'json'
    
    for i in tqdm(range(len(data_test))):
        file_path = f'{folder_path}/{str(i)}.json'
        if os.path.exists(file_path) and not args.overwrite:
            continue

        sample = data_test[i]
        story_id = process_id(args.dataset, sample['id'])
        pred_TG = pred_TGs.get(story_id, None)
        if pred_TG is None and not args.no_TG:
            continue

        cur_prompt = my_generate_prompt_TG_Reasoning(args.dataset, '', sample['story'], pred_TG, 
                                                     sample['external knowledge'], sample['question'], 
                                                     None, None, args.ICL, Q_type=sample['Q-Type'], 
                                                     mode='test', f_no_TG=args.no_TG, 
                                                     prompt_format=args.prompt_format)

        file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
        if not os.path.exists(file_path_past_res):
            continue

        with open(file_path_past_res) as json_file:
            past_res = json.load(json_file)
        CoT, _ = parse_TGR_pred(past_res['prediction'], args.prompt_format)
        if CoT is None:
            continue

        cur_prompt += f'{{\n"Thought": {json.dumps(CoT)},\n"Answer":' if using_json else f'\nThought: {CoT}\n\nAnswer:'

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_ppl(peft_model, tokenizer, input_prompts, samples, file_paths, using_json)
            input_prompts, file_paths, samples = [], [], []

    if input_prompts:
        run_one_batch_ppl(peft_model, tokenizer, input_prompts, samples, file_paths, using_json)


def main():
    args = parse_arguments()
    data_test, pred_TGs, dataset_name, split_name = prepare_data(args)

    if args.print_prompt:
        for i in range(5):
            print_prompt_sample(data_test[i], pred_TGs, dataset_name, args)
        
    strategy = 'TGR' if not args.no_TG else 'storyR'
    peft_model_id = f"../model_weights/{dataset_name}_{strategy}{split_name}/final"

    peft_model, tokenizer = load_model_and_tokenizer(model_path, peft_model_id)
    folder_path, folder_path_past_res = setup_directories(dataset_name, strategy, split_name)

    evaluate(peft_model, tokenizer, data_test, pred_TGs, args, folder_path, folder_path_past_res)


if __name__ == "__main__":
    main()