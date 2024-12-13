import torch
import time
import json
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer



def write_pretty_json(file_path, data):
    """
    Write data to a JSON file with pretty formatting.
    """
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def prepare_prompts(prompts, tokenizer, batch_size=16):
    """
    Batch, left pad (for inference), and tokenize prompts.
    """
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
        )
    tokenizer.padding_side = "right"
    return batches_tok


def run_inference(model, tokenizer, prompts, accelerator, batch_size=16, max_new_tokens=200):
    """
    Run inference on the prompts across multiple GPUs using accelerate.
    """
    results = dict(outputs=[], num_tokens=0)

    # Divide the prompts list onto the available GPUs
    with accelerator.split_between_processes(prompts) as split_prompts:
        # Prepare batched and tokenized prompts
        prompt_batches = prepare_prompts(split_prompts, tokenizer, batch_size=batch_size)
        
        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=max_new_tokens)
            
            # Remove prompt tokens from generated outputs
            outputs_tokenized = [
                tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)
            ] 
            
            # Count and decode generated tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizer.batch_decode(outputs_tokenized)
            
            # Store results
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
        
    return [results]  # Transform to list for gather_object()


def main():
    # Initialize Accelerator
    accelerator = Accelerator()

    # Prompts
    prompts_all = [
        "The King is dead. Long live the Queen.",
        "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
        "The story so far: in the beginning, the universe was created.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "The sweat wis lashing oafay Sick Boy; he wis trembling.",
        "124 was spiteful. Full of Baby's venom.",
        "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
        "I write this sitting in the kitchen sink.",
        "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
    ] * 10

    # Load model and tokenizer
    model_path = "meta-llama/Llama-2-13b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # Synchronize GPUs and start timer
    accelerator.wait_for_everyone()
    start = time.time()

    # Run inference
    results_gathered = gather_object(run_inference(model, tokenizer, prompts_all, accelerator))

    # Process and display results
    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])
        print(f"tokens/sec: {num_tokens // timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")


if __name__ == "__main__":
    main()