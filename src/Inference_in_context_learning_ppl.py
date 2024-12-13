import sys
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from utlis import *
from Models import *
from prompt_generation import *


os.environ["WANDB_DISABLED"] = "true"

# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help="Dataset to use for evaluation.")
    parser.add_argument('--model', type=str, help="Model to use for inference (e.g., Llama2-7b, Llama2-13b).")
    parser.add_argument('--CoT', action='store_true', help="Enable Chain-of-Thought (CoT) reasoning.")
    parser.add_argument('--ICL', action='store_true', help="Enable in-context learning during testing.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing results.")
    parser.add_argument('--shorten_story', action='store_true', help="Shorten input stories for models with limited context.")
    parser.add_argument('--print_prompt', action='store_true', help="Print example prompts for inspection.")
    parser.add_argument('--unit_test', action='store_true', help="Run the script in unit test mode with a small subset of data.")

    return parser.parse_args()


def configure_experiment(args):
    """
    Configures the experiment based on input arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: A dictionary containing configuration parameters for the experiment.
    """
    dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
    model_selection = ['Llama2-7b', 'Llama2-13b', 'Llama2-70b'].index(args.model)
    
    config = {
        'dataset_name': ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection],
        'split_name': ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection],
        'prefix': ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection],
        'model_name': ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection],
        'learning_setting': 'SP' if not args.CoT else 'CoT',
        'f_ICL': args.ICL,
        'f_overwrite': args.overwrite,
        'f_shorten_story': args.shorten_story,
        'f_print_prompt': args.print_prompt,
        'f_unit_test': args.unit_test,
    }
    return config


def load_data(config, args):
    """
    Loads the test dataset based on the configuration.

    Args:
        config (dict): Experiment configuration.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dataset: A Hugging Face dataset object containing the test data.
    """
    dataset = load_dataset(dataset_path, f'{config["dataset_name"]}_TGR')
    data_test = dataset[config['prefix'] + 'test']
    if args.unit_test:
        data_test = create_subset(data_test, 10)
    return data_test


def initialize_model(config):
    """
    Initializes the causal language model and tokenizer for inference.

    Args:
        config (dict): Experiment configuration.

    Returns:
        tuple: A tuple containing the model and tokenizer objects.
    """
    model_name_cmp = f'meta-llama/{config["model_name"]}'
    tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_name_cmp, load_in_8bit=True, device_map="auto")
    model.eval()
    return model, tokenizer


def process_prompts(args, data_test, config, model, tokenizer):
    """
    Processes the input data, generates prompts, and runs inference.

    Args:
        args (argparse.Namespace): Parsed arguments.
        data_test (Dataset): The test dataset.
        config (dict): Experiment configuration.
        model (AutoModelForCausalLM): The initialized language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
    """
    # Create output folder
    folder_path = f'../results/{config["dataset_name"]}_ICL_{config["learning_setting"]}{config["split_name"]}_{config["model_name"]}_ppl' \
                    if config["f_ICL"] else f'../results/{config["dataset_name"]}{config["split_name"]}_{config["model_name"]}_ppl'
    os.makedirs(folder_path, exist_ok=True)

    batch_size = 4
    input_prompts, file_paths, samples = [], [], []

    for i in tqdm(range(len(data_test)), desc="Processing Prompts"):
        file_path = f'{folder_path}/{str(i)}.json'
        if os.path.exists(file_path) and not config["f_overwrite"]:
            continue

        # Generate prompt
        sample = data_test[i]
        cur_prompt = my_generate_prompt_ICL(
            config["dataset_name"], config["split_name"], config["learning_setting"],
            sample['story'], sample['question'], sample['candidates'],
            config["f_ICL"], config["f_shorten_story"], args.CoT, Q_type=sample['Q-Type']
        )

        # Append Chain-of-Thought (if enabled)
        if args.CoT:
            folder_path_past_res = folder_path.replace("_ppl", "")
            file_path_past_res = f'{folder_path_past_res}/{str(i)}.json'
            if not os.path.exists(file_path_past_res):
                continue
            with open(file_path_past_res) as json_file:
                past_res = json.load(json_file)
            cur_prompt += process_CoT(past_res['prediction'])

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        # Run batch inference
        if len(input_prompts) >= batch_size:
            run_one_batch_ppl(model, tokenizer, input_prompts, samples, file_paths, using_json=False)
            input_prompts, file_paths, samples = [], [], []

    # Process remaining prompts
    if input_prompts:
        run_one_batch_ppl(model, tokenizer, input_prompts, samples, file_paths, using_json=False)


def main():
    """
    The main function that orchestrates the experiment workflow.
    It parses arguments, loads the dataset, initializes the model,
    processes prompts, and runs inference.
    """
    args = parse_arguments()
    config = configure_experiment(args)
    
    print("Loading Dataset...")
    data_test = load_data(config, args)
    print(data_test)

    if config["f_print_prompt"]:
        for i in range(5):
            sample = data_test[i]
            prompt = my_generate_prompt_ICL(
                config["dataset_name"], config["split_name"], config["learning_setting"],
                sample['story'], sample['question'], sample['candidates'],
                config["f_ICL"], config["f_shorten_story"], args.CoT, Q_type=sample['Q-Type']
            )
            print(prompt)
            print('===============================')
    
    print("Initializing Model...")
    model, tokenizer = initialize_model(config)

    print("Processing Prompts...")
    process_prompts(args, data_test, config, model, tokenizer)


if __name__ == '__main__':
    main()