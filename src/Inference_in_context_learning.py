import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse


os.environ["WANDB_DISABLED"] = "true"

# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--CoT', action='store_true', help="Enable Chain-of-Thought (CoT) reasoning.")
    parser.add_argument('--ICL', action='store_true', help="Enable in-context learning during testing.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing results.")
    parser.add_argument('--shorten_story', action='store_true', help="Shorten input stories for models with limited context.")
    parser.add_argument('--print_prompt', action='store_true', help="Print example prompts for inspection.")
    parser.add_argument('--unit_test', action='store_true', help="Run the script in unit test mode with a small subset of data.")
    return parser.parse_args()


def load_test_data(args):
    """Loads the test dataset based on the provided arguments."""
    dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
    split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
    prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]
    
    dataset = load_dataset(dataset_path, f'{dataset_name}_TGR')
    data_test = dataset[prefix + 'test']
    
    if args.unit_test:
        data_test = create_subset(data_test, 10)
    
    return data_test, dataset_name, split_name


def initialize_model_and_tokenizer(model_name):
    """Initializes the model and tokenizer based on the model name."""
    model = None
    tokenizer = None
    if 'Llama' in model_name:
        model_name_cmp = f'meta-llama/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name_cmp, load_in_8bit=True, device_map="auto"
        )
        model.eval()
    return model, tokenizer


def generate_and_save_prompts(data_test, model, tokenizer, folder_path, args, dataset_name, split_name):
    """Generates prompts, processes them in batches, and saves results."""
    batch_size = 4
    input_prompts = []
    file_paths = []
    samples = []
    
    for i in tqdm(range(len(data_test))):
        file_path = f'{folder_path}/{str(i)}.json'
        if os.path.exists(file_path) and (not args.overwrite):
            continue

        sample = data_test[i]
        cur_prompt = my_generate_prompt_ICL(
            dataset_name, split_name, 'CoT' if args.CoT else 'SP', 
            sample['story'], sample['question'], sample['candidates'],
            args.ICL, args.shorten_story, args.CoT, Q_type=sample['Q-Type']
        )
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_ICL(args.model, model, tokenizer, input_prompts, samples, file_paths)
            input_prompts, file_paths, samples = [], [], []

    if len(input_prompts) > 0:
        run_one_batch_ICL(args.model, model, tokenizer, input_prompts, samples, file_paths)


def main():
    args = parse_args()

    # Load dataset and initialize variables
    data_test, dataset_name, split_name = load_test_data(args)
    model_selection = ['gpt-3.5', 'gpt-4', 'Llama2-7b', 'Llama2-13b', 'Llama2-70b'].index(args.model)
    model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]
    
    folder_path = f'../results/{dataset_name}_ICL_{{"CoT" if args.CoT else "SP"}}{split_name}_{model_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # Print example prompts if specified
    if args.print_prompt:
        for i in range(5):
            sample = data_test[i]
            prompt = my_generate_prompt_ICL(
                dataset_name, split_name, 'CoT' if args.CoT else 'SP',
                sample['story'], sample['question'], sample['candidates'],
                args.ICL, args.shorten_story, args.CoT, Q_type=sample['Q-Type']
            )
            print(prompt)
            print('===============================')

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    # Generate and save prompts
    generate_and_save_prompts(data_test, model, tokenizer, folder_path, args, dataset_name, split_name)


if __name__ == "__main__":
    main()