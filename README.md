## [ACL 24 (main)] TG-LLM: Large Language Models Can Learn Temporal Reasoning

This repository contains the code for the paper [Large Language Models Can Learn Temporal Reasoning](https://arxiv.org/pdf/2401.06853.pdf).

Our framework (TG-LLM) performs **temporal reasoning** in two steps: 1) **Text-to-Temporal Graph translation:** generate (relevant) temporal graph given the context and keyword (extracted from questions); 2) **Temporal Graph Reasoning:** perform Chain-of-Thought reasoning over the temporal graph.

<br>

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/TG-LLM/main/misc/Framework.png' width=550>
</p>




## Quick Start

We use [Hugging Face](https://huggingface.co/) platform to load the Llama2 model family. Make sure you have an account ([Guidance](https://huggingface.co/blog/llama2)).

The structure of the file folder should be like
```sh
TG-LLM/
│
├── materials/
│
├── model_weights/
│
├── results/
│
└── src/
```

<h4> Preparation: </h4>

```sh
# git clone this repo

# create a new environment with anaconda and install the necessary Python packages

# install hugging face packages to load Llama2 models and datasets

# create the folders
cd TG-LLM
mkdir model_weights
mkdir results
cd src
```

<h4> For our TG-LLM framework: </h4>

- Step 1: text-to-temporal graph translation

```sh
# Train and test on TGQA dataset
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TGQA --train --print_prompt
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TGQA --test --ICL --print_prompt

# Train and test on TimeQA dataset (Since some stories in TimeQA and TempReason are too long to feed into Llama2 (max_context_len: 4096), it is recommended to shorten the story.)
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TimeQA --train --print_prompt --shorten_story
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TimeQA --test --ICL --print_prompt --shorten_story

# Train on TGQA, test on TimeQA
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TGQA --train --transferred_dataset TimeQA --print_prompt
python SFT_with_LoRA_text_to_TG_Trans.py --dataset TimeQA --test --shorten_story --ICL --print_prompt --transferred
```

- Step 2: temporal graph reasoning

```sh
# Obtain CoT sampling prob
python CoT_bootstrap.py --dataset TGQA --print_prompt

# Train and test on TGQA dataset
python SFT_with_LoRA_TG_Reasoning.py --dataset TGQA --train --CoT_bs --data_aug --print_prompt
python SFT_with_LoRA_TG_Reasoning.py --dataset TGQA --test --ICL --print_prompt

# To obtain inference results based on perplexity
python SFT_with_LoRA_TG_Reasoning_ppl.py --dataset TGQA --ICL --print_prompt
```

<h4> For other leading LLMs (GPT series/Llama2 family): </h4>

- Use in-context learning only

```sh
# Test on TGQA with Llama2-13b with ICL only
python Inference_in_context_learning.py --dataset TGQA --model Llama2-13b --CoT --ICL --print_prompt

# To obtain inference results based on perplexity
python Inference_in_context_learning_ppl.py --dataset TGQA --model Llama2-13b --CoT --ICL --print_prompt
```

- Use SFT with vanilla CoT (story, question, CoT, answer)
```sh
# Train and test on TGQA dataset
python SFT_with_LoRA_TG_Reasoning.py --dataset TGQA --train --print_prompt --no_TG
python SFT_with_LoRA_TG_Reasoning.py --dataset TGQA --test --ICL --print_prompt --no_TG

# To obtain inference results based on perplexity
python SFT_with_LoRA_TG_Reasoning_ppl.py --dataset TGQA --ICL --print_prompt --no_TG
```


<h4> For evaluation: </h4>

```sh
# To evaluate our framework
python Evaluation.py --dataset TGQA --model Llama2-13b --SFT

# To evaluate other leading LLMs with ICL only
python Evaluation.py --dataset TGQA --model Llama2-13b --ICL_only --CoT

# To evaluate other leading LLMs with SFT on vanilla CoT
python Evaluation.py --dataset TGQA --model Llama2-13b --SFT --no_TG
```

## Prompt Format
We have changed the prompt format of our framework into **JSON** which is much eaiser to parse and doesn't hurt the peformance.

## Datasets

All the datasets (TGQA, TimeQA, TempReason) can be found [here](https://huggingface.co/datasets/sxiong/TGQA).

To download the dataset, install [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart) and then use the following command:

```python
from datasets import load_dataset
dataset = load_dataset("sxiong/TGQA", "TGQA_Story_TG_Trans") # Six configs available: "TGQA_Story_TG_Trans", "TGQA_TGR", "TempReason_Story_TG_Trans", "TempReason_TGR", "TimeQA_Story_TG_Trans", "TimeQA_TGR"
print(dataset) # Print dataset to see the statistics and available splits
split = dataset['train']  # Multiple splits available: "train", "val", "test"
```


## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
@misc{xiong2024large,
      title={Large Language Models Can Learn Temporal Reasoning}, 
      author={Siheng Xiong and Ali Payani and Ramana Kompella and Faramarz Fekri},
      year={2024},
      eprint={2401.06853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
