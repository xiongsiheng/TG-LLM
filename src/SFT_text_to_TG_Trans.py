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
import wandb


# You can change into local paths if they are downloaded locally
dataset_path = "sxiong/TGQA" 
model_path = "meta-llama/Llama-2-13b-hf"




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Name of the dataset to use. Options: 'TGQA', 'TimeQA', 'TempReason'.")
    parser.add_argument('--train', action='store_true', help="Flag to enable training mode.")
    parser.add_argument('--test', action='store_true', help="Flag to enable testing mode.")
    parser.add_argument('--overwrite', action='store_true', help="Flag to overwrite existing results.")
    parser.add_argument('--ICL', action='store_true', help="Flag to enable in-context learning during testing.")
    parser.add_argument('--shorten_story', action='store_true', help="Flag to shorten the input story if it is too long.")
    parser.add_argument('--hard_mode', action='store_true', help="Enable hard mode: only relations provided.")
    parser.add_argument('--print_prompt', action='store_true', help="Flag to print example prompts.")
    parser.add_argument('--unit_test', action='store_true', help="Flag to enable unit testing with small data subsets.")
    parser.add_argument('--transferred_dataset', type=str, help="Name of the transferred dataset for testing.")
    parser.add_argument('--transferred', action='store_true', help="Flag to enable transfer learning mode.")
    parser.add_argument('--resume_from', type=str, help="Path to resume training from a checkpoint.")
    parser.add_argument('--prompt_format', type=str, default='plain', help="Prompt format: 'plain' or 'json'.")
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
            project=f"{args.dataset}_text_to_TG_Trans"
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def load_and_prepare_dataset(dataset_name, f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, prompt_format):
    """
    Load and preprocess the dataset by generating prompts for each sample.
    Args:
        dataset_name (str): Name of the dataset to load.
        f_ICL (bool): Flag for in-context learning during testing.
        f_shorten_story (bool): Flag to shorten the story input.
        f_hard_mode (bool): Enable hard mode with relations only.
        transferred_dataset_name (str): Transferred dataset name, if applicable.
        prompt_format (str): Format for the prompts ('plain' or 'json').
    Returns:
        tuple: Preprocessed train, validation, and test datasets.
    """
    dataset = load_dataset(dataset_path, f'{dataset_name}_Story_TG_Trans')
    data_train = dataset['train']
    data_val = dataset['val']
    data_test = dataset['test']

    def add_prompt(sample):
        sample['prompt'] = my_generate_prompt_TG_trans(
            dataset_name, sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times'],
            f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, max_story_len=1200, prompt_format=prompt_format
        )
        return sample

    data_train = data_train.map(add_prompt)
    data_val = data_val.map(add_prompt)
    return data_train, data_val, data_test


def print_example_prompts(data, mode, prompt_format, **kwargs):
    """
    Print example prompts for debugging or inspection.
    Args:
        data (Dataset): Dataset to sample prompts from.
        mode (str): Mode of operation ('train' or 'test').
        prompt_format (str): Format for the prompts ('plain' or 'json').
        kwargs: Additional arguments (dataset_name, flags, etc.).
    """
    for i in range(5):
        sample = data[i]
        TG = sample['TG'] if mode == 'train' else None
        eos_token = '</s>' if mode == 'train' else ''
        prompt = my_generate_prompt_TG_trans(
            kwargs['dataset_name'], sample['story'], TG, sample['entities'],
            sample['relation'], sample['times'], kwargs['f_ICL'], kwargs['f_shorten_story'],
            kwargs['f_hard_mode'], kwargs['transferred_dataset_name'], mode=mode, 
            eos_token=eos_token, 
            prompt_format=prompt_format
        )
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


def train_model(model, tokenizer, data_train, data_val, dataset_name, transferred_dataset_name, f_unit_test, resume_from_checkpoint):
    """
    Fine-tune the model on the training dataset.
    Args:
        model (AutoModelForCausalLM): Pretrained causal language model.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        data_train (Dataset): Training dataset.
        data_val (Dataset): Validation dataset.
        dataset_name (str): Name of the dataset being used.
        transferred_dataset_name (str): Name of the transferred dataset, if applicable.
        f_unit_test (bool): Flag for unit testing mode.
        resume_from_checkpoint (str): Path to checkpoint for resuming training.
    """
    def formatting_func(sample):
        return [p for p in sample['prompt']]

    output_dir = f"../model_weights/{dataset_name}_story_TG_trans"
    if transferred_dataset_name:
        output_dir = f"../model_weights/{dataset_name}_to_{transferred_dataset_name}_story_TG_trans"

    max_steps = 5 if f_unit_test else -1  # early stop by setting max_steps
    response_template = "### Output"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    batch_size = 12 if dataset_name == 'TGQA' else 4
    max_seq_length = 1500 if dataset_name == 'TGQA' else 4096

    SFT_with_LoRA(
        model, tokenizer, output_dir, formatting_func, data_train, data_val,
        batch_size, max_seq_length, max_steps, resume_from_checkpoint=resume_from_checkpoint, collator=collator
    )


def test_model(model, tokenizer, data_test, folder_path, **kwargs):
    """
    Run inference on the test dataset and save outputs.
    Args:
        model (AutoModelForCausalLM): Base causal language model.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        data_test (Dataset): Test dataset.
        folder_path (str): Path to store the output results.
        kwargs: Additional configurations (batch size, max tokens, etc.).
    """
    peft_model = PeftModel.from_pretrained(
        model, kwargs['peft_model_id'], torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp"
    )
    peft_model.eval()
    
    if kwargs['prompt_format'].lower() == 'json':
        identifier='{\n"Timeline":'
    else:
        identifier='Timeline:'

    input_prompts, file_paths, samples = [], [], []
    for i in tqdm(range(len(data_test))):
        file_path = f"{folder_path}/{str(i)}.json"
        if os.path.exists(file_path) and not kwargs['f_overwrite']:
            continue

        sample = data_test[i]
        cur_prompt = my_generate_prompt_TG_trans(
            kwargs['dataset_name'], sample['story'], None, sample['entities'],
            sample['relation'], sample['times'], kwargs['f_ICL'], kwargs['f_shorten_story'],
            kwargs['f_hard_mode'], kwargs['transferred_dataset_name'], mode='test',
            eos_token='', prompt_format=kwargs['prompt_format']
        )
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= kwargs['batch_size']:
            run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=kwargs['max_new_tokens'], 
                                     prompt_format=kwargs['prompt_format'], identifier=identifier)
            input_prompts, file_paths, samples = [], [], []

    if len(input_prompts) > 0:
        run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=kwargs['max_new_tokens'], 
                                 prompt_format=kwargs['prompt_format'], identifier=identifier)


def main():
    args = parse_arguments()
    data_train, data_val, data_test = load_and_prepare_dataset(
        args.dataset, args.ICL, args.shorten_story, args.hard_mode, args.transferred_dataset, args.prompt_format
    )

    if args.print_prompt:
        print_example_prompts(data_train if args.train else data_test, 'train' if args.train else 'test', args.prompt_format,
                              dataset_name=args.dataset, f_ICL=args.ICL, f_shorten_story=args.shorten_story,
                              f_hard_mode=args.hard_mode, transferred_dataset_name=args.transferred_dataset)

    setup_wandb(args)

    model, tokenizer = load_model_and_tokenizer()

    if args.train:
        train_model(model, tokenizer, data_train, data_val, args.dataset, args.transferred_dataset, args.unit_test, args.resume_from)

    if args.test:
        folder_path = f"../results/{args.dataset}_story_TG_trans"
        if args.transferred:
            folder_path = f"../results/TGQA_to_{args.dataset}_story_TG_trans"
        os.makedirs(folder_path, exist_ok=True)

        test_model(
            model, tokenizer, data_test, folder_path, dataset_name=args.dataset,
            peft_model_id=f"../model_weights/{args.dataset}_story_TG_trans/final", f_ICL=args.ICL,
            f_shorten_story=args.shorten_story, f_hard_mode=args.hard_mode,
            transferred_dataset_name=args.transferred_dataset, f_overwrite=args.overwrite,
            prompt_format=args.prompt_format, batch_size=1, max_new_tokens=1024 if args.dataset == 'TGQA' else 512
        )


if __name__ == "__main__":
    main()