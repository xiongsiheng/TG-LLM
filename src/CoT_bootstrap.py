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

# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 
model_path = "meta-llama/Llama-2-13b-hf"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--print_prompt', action='store_true', help='whether to print the example prompt for the model')
    parser.add_argument('--overwrite', action='store_true', help='whether overwrite existing results')      
    parser.add_argument('--unit_test', action='store_true', help='whether to run the unit test (only for debugging)')
    parser.add_argument('--prompt_format', type=str, default='plain', help='whether use plain (text) or json as prompt format')
    return parser.parse_args()


def load_and_prepare_data(dataset_name, dataset_selection, unit_test):
    """
    Load and prepare the training and validation datasets.
    Args:
        dataset_name (list): List of dataset names.
        dataset_selection (int): Selected index for the dataset.
        unit_test (bool): Whether to run in unit test mode.
    Returns:
        tuple: Training and validation datasets.
    """
    dataset = load_dataset(dataset_path, f'{dataset_name}_TGR')
    split_train = ['train', 'easy_train', 'hard_train', 'l2_train', 'l3_train'][dataset_selection]
    split_val = ['val', 'easy_val', 'hard_val', 'l2_val', 'l3_val'][dataset_selection]
    
    data_train = dataset[split_train]
    data_val = dataset[split_val]
    
    if unit_test:
        data_train = create_subset(data_train, 10)
        data_val = create_subset(data_val, 10)

    return data_train, data_val, split_train, split_val


def setup_model(model_name):
    """
    Load and configure the tokenizer and model.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tuple: Tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def print_example_prompts(data_train, prompt_format):
    """
    Print example prompts from the training data.
    Args:
        data_train (Dataset): Training data.
        prompt_format (str): Format for the prompt.
    """
    for i in range(5):
        sample = data_train[i]
        prompt = my_generate_prompt_CoT_bs(sample['TG'], sample['external knowledge'], sample['question'], prompt_format=prompt_format)
        print(prompt)
        print('===============================')


def CoT_bootstrap(data, filename, model, tokenizer, dataset_name, overwrite, batch_size=4, prompt_format='plain'):
    """
    Perform Chain-of-Thought (CoT) bootstrap on the dataset.
    Args:
        data: Dataset to process.
        filename: Output filename to save results.
        model: Pretrained model.
        tokenizer: Pretrained tokenizer.
        dataset_name: Name of the dataset.
        overwrite: Whether to overwrite existing results.
        batch_size: Batch size for processing.
        prompt_format: Format of the prompts.
    """
    folder_path = f'../results/{dataset_name}_TGR_CoT_bs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = f'{folder_path}/{filename}'
    input_prompts = []
    input_samples = []

    for sample in tqdm(data):
        cur_id = sample['id']
        cur_file_path = f'{file_path}_{cur_id}.json'
        if os.path.exists(cur_file_path) and (not overwrite):
            continue

        cur_prompt = my_generate_prompt_CoT_bs(sample['TG'], sample['external knowledge'], sample['question'], prompt_format=prompt_format)
        input_prompts.append(cur_prompt)
        input_samples.append(sample)

        if len(input_prompts) >= batch_size:
            run_one_batch_CoT_bs(model, tokenizer, input_prompts, input_samples, file_path, prompt_format=prompt_format)
            input_prompts = []
            input_samples = []

    # Last batch
    if len(input_prompts) > 0:
        run_one_batch_CoT_bs(model, tokenizer, input_prompts, input_samples, file_path, prompt_format=prompt_format)


def main():
    # Parse arguments
    args = parse_arguments()

    # Configurations
    dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]

    # Load datasets
    data_train, data_val, split_train, split_val = load_and_prepare_data(dataset_name, dataset_selection, args.unit_test)

    # Print example prompts if enabled
    if args.print_prompt:
        print_example_prompts(data_train, args.prompt_format)

    # Setup model and tokenizer
    tokenizer, model = setup_model(model_path)

    # Run CoT bootstrap on train and validation datasets
    CoT_bootstrap(data_train, split_train, model, tokenizer, dataset_name, args.overwrite, prompt_format=args.prompt_format)
    CoT_bootstrap(data_val, split_val, model, tokenizer, dataset_name, args.overwrite, prompt_format=args.prompt_format)


if __name__ == "__main__":
    main()