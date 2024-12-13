import sys
import torch
import json
import numpy as np
import itertools
from utlis import *

import transformers
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer

import torch
import openai
import time




openai.api_key = None  # add your own api_key here (you can look at the openai website to get one)





def my_gpt_completion(openai_model, messages, timeout, max_new_tokens=128, wait_time=0):
    '''
    Use the GPT model to generate the completion
    We can add exception handling here to avoid interruption

    args:
    openai_model: the model name
    messages: the messages in the conversation
    timeout: the timeout for the request
    max_new_tokens: the maximum new tokens for the completion
    wait_time: the wait time between requests

    return:
    response: the generated response, str
    '''
    completion = openai.ChatCompletion.create(model=openai_model,
                                              messages=messages,
                                              request_timeout = timeout,
                                              temperature=0.01,
                                              max_tokens=max_new_tokens
                                            )
    response = completion['choices'][0]["message"]["content"]
    time.sleep(wait_time) # wait for a while to avoid the rate limit

    return response


def run_one_batch_ICL(model_name, model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=512):
    '''
    Generate the completion for one batch of input prompts

    args:
    input_prompts: the input prompts, list
    samples: the samples, list
    file_paths: the file paths to save the results, list
    max_new_tokens: the maximum new tokens for the completion

    return:
    None
    '''
    if 'Llama' in model_name:
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.01,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
              )


        for j in range(len(input_prompts)):
            op = tokenizer.decode(generation_output[j], skip_special_tokens=True)
            op = op[len(input_prompts[j]):]
            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    if 'gpt' in model_name:
        for j in range(len(input_prompts)):
            messages = []
            messages.append({"role": "user", "content": input_prompts[j]})
            op = my_gpt_completion(model_name, messages, 600, max_new_tokens=max_new_tokens, wait_time=0.5)

            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)


    return


def run_one_batch_CoT_bs(model, tokenizer, input_prompts, samples, file_path, prompt_format='plain'):
    '''
    For each sample, calculate the contrastive score for each CoT. Then save the results to the corresponding files.
    
    Args:
    model: the model
    tokenizer: the tokenizer
    input_prompts: the input prompts, list
    samples: the samples, dict
    file_path: the file path to save the results, str
    prompt_format: the format of the prompt, str

    Returns:
    samples: the samples with the CoT sample probability, dict
    '''
    assert prompt_format.lower() in ['plain', 'json'], "Prompt format is not recognized."
    
    gamma = 0.5    # score = logProbs_pos + gamma*(logProbs_pos - logProbs_neg)
    for j in range(len(input_prompts)):
        cur_sample = samples[j]

        # Prepare the combinations of CoT and candidates
        neg_ans = [cand for cand in cur_sample['candidates'] if cand not in cur_sample['answer']]
        combinations = list(itertools.product(cur_sample['CoT'], neg_ans + cur_sample['answer']))
        cur_prompts = []
        context_len = []
        for comb in combinations:
            CoT = comb[0]
            # CoT = CoT.replace('\n', ' ')
            context = input_prompts[j] + f'{{\n"Thought": {json.dumps(CoT)},\n"Answer": ' if prompt_format.lower() == 'json' else \
                    input_prompts[j] + f'\nThought: {CoT}\n\nAnswer: '
            final = context + f'{json.dumps([comb[1]])}\n}}```' if prompt_format.lower() == 'json' else context + comb[1]

            len_bf = tokenizer(context, return_tensors="pt")["input_ids"].shape[1]
            len_af = tokenizer(final, return_tensors="pt")["input_ids"].shape[1]
            
            cur_prompts.append(final)
            
            # The length of the context should be at least 1 less than the length of all the tokens
            context_len.append(min(len_bf, len_af-1))
        
        loss_per_answer = obtain_loss_in_batch(model, tokenizer, cur_prompts, context_len)

        # Split the losses back to individual CoTs
        loss_per_answer = loss_per_answer.reshape((len(cur_sample['CoT']), -1))
        logProbs_pos = np.mean(loss_per_answer[:, len(neg_ans):], axis=1)
        logProbs_neg = np.mean(loss_per_answer[:, :len(neg_ans)], axis=1)

        # Constrastive score:
        scores = logProbs_pos + gamma*(logProbs_pos - logProbs_neg)
        scores = [np.exp(-10*s) for s in scores]

        # Normalize the scores to get the probability
        cur_sample['CoT_sample_prob'] = (scores/np.sum(scores)).tolist()
        cur_id = cur_sample['id']
        cur_file_path = f'{file_path}_{cur_id}.json'
        with open(cur_file_path, 'w') as json_file:
            json.dump(cur_sample, json_file)

    return 



def run_one_batch_ppl(model, tokenizer, input_prompts, samples, file_paths, using_json=True):
    '''
    Given a batch of input prompts and candidates, calculate the perplexity of the candidates and choose the best one. Then save the results to the corresponding files.

    Args:
    model: the model
    tokenizer: the tokenizer
    input_prompts: the input prompts
    samples: the samples
    file_paths: the file paths to save the results
    using_json: whether to use json format

    Return:
    None
    '''

    for j in range(len(input_prompts)):
        cur_sample = samples[j]

        # Prepare the combinations of CoT and candidates
        cand = cur_sample['candidates'] + cur_sample['answer']
        combinations = list(itertools.product([input_prompts[j]], cand))
        cur_prompts = []
        context_len = []
        for comb in combinations:
            context = comb[0]
            if using_json:
                final = context + f'{json.dumps([comb[1]])}\n}}```'
            else:
                final = context + comb[1]
            cur_prompts.append(final)
            
            len_bf = tokenizer(context, return_tensors="pt")["input_ids"].shape[1]
            len_af = tokenizer(final, return_tensors="pt")["input_ids"].shape[1]
            
            # The length of the context should be at least 1 less than the length of all the tokens
            context_len.append(min(len_bf, len_af-1))
        
        loss_per_answer = obtain_loss_in_batch(model, tokenizer, cur_prompts, context_len)

        op = cand[np.argmin(loss_per_answer.reshape(-1))]
        cur_sample.update({'prediction': op})

        with open(file_paths[j], 'w') as json_file:
            json.dump(cur_sample, json_file)

    return


def run_one_batch_generation(model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=512, prompt_format='plain', identifier=None):
    '''
    Generate the predictions for one batch of samples and save the results.

    args:
        model: model, the model
        tokenizer: tokenizer
        input_prompts: list of strings, input prompts
        samples: list of dictionaries, samples
        file_paths: list of strings, file paths
        max_new_tokens: int, maximum number of new tokens
        prompt_format: str, the format of the prompt

    return:
        None
    '''
    assert prompt_format.lower() in ['plain', 'json'], "Prompt format is not recognized."

    input_prompts_input = input_prompts
    if identifier is not None:
        input_prompts_input = [f"{prompt}{identifier}" for prompt in input_prompts]
    
    input_tokens = tokenizer(input_prompts_input, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

    with torch.cuda.amp.autocast():
        generation_output = model.generate(
            input_ids=input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.01,
            repetition_penalty=1.15,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            )

    for j in range(len(input_prompts)):
        op = tokenizer.decode(generation_output[j], skip_special_tokens=True)
        if prompt_format == 'json':
            op = op[len(input_prompts[j]) - len('\n```json'):]
        else:
            op = op[len(input_prompts[j]):]
        cur_sample = samples[j]
        cur_sample.update({'prediction': op})

        with open(file_paths[j], 'w') as json_file:
            json.dump(cur_sample, json_file)

    return


def SFT_with_LoRA(model, tokenizer, output_dir, formatting_func, data_train, data_val, batch_size, max_seq_length, max_steps, resume_from_checkpoint=None,
                  collator=None):
    # lora config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Loading in 8 bit ..."
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    per_device_train_batch_size = batch_size
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = batch_size
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 30
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    evaluation_strategy = "epoch"
    lr_scheduler_type = "cosine"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
                resume_from_checkpoint=resume_from_checkpoint
            )

    # SFT with lora
    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(f"{output_dir}/final")
    return


def obtain_loss_in_batch(model, tokenizer, cur_prompts, context_len):
    '''
    Given a batch of prompts, calculate the loss for each answer in the batch.
    
    Args:
    model: the model
    tokenizer: the tokenizer
    cur_prompts: the current prompts, list
    context_len: the length of the context, list

    Returns:
    loss_per_answer: the loss for each answer, list
    '''
    # Tokenize the entire batch of answers at once with truncation
    input_tokens = tokenizer(cur_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"].to("cuda")        

    # Create target_ids with masked context
    target_ids = input_tokens.clone()
    for i in range(len(context_len)):
        target_ids[i, :context_len[i]] = -100  # mask the context before the answer

    # Mask padding tokens
    padding_mask = input_tokens == tokenizer.pad_token_id
    target_ids[padding_mask] = -100  # mask the padding tokens

    # # Verify target_ids
    # print("Target IDs after padding mask:", target_ids)

    # Process the batch
    with torch.no_grad():
        outputs = model(input_tokens, labels=target_ids)
        logits = outputs.logits

    # Calculate loss per token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Reshape loss to match input tokens shape
    loss = loss.view(shift_labels.size())

    # Mask loss for padding tokens
    loss[padding_mask[:, 1:]] = 0.0

    # # Verify loss tensor
    # print("Loss tensor:", loss)

    # Aggregate loss for each answer
    valid_counts = (loss != 0).sum(dim=1)
    valid_counts[valid_counts == 0] = 1  # Avoid division by zero
    loss_per_answer = loss.sum(dim=1) / valid_counts
    loss_per_answer = loss_per_answer.cpu().numpy()

    return loss_per_answer