## TG-LLM: Large Language Models Can Learn Temporal Reasoning

This repository contains the code for the paper [Large Language Models Can Learn Temporal Reasoning](https://arxiv.org/pdf/2401.06853.pdf).

Our framework:

<p align="center">
  <img src='https://github.com/xiongsiheng/TG-LLM/blob/main/misc/Framework.png' width=400>
</p>

# How to run

We use [Hugging Face](https://huggingface.co/) platform to load the Llama2 model family. Make sure you have an account ([Guidance](https://huggingface.co/blog/llama2)).

The structure of the file folder should be like
```sh
TG-LLM/
│
├── src/
│
├── dataset/
│
├── model_weights/
│
├── materials/
│
└── results/
```

For our TG-LLM framework

```sh
cd src
python SFT_with_LoRA_text_to_TG_Trans.py
python CoT_bootstrap.py
python SFT_with_LoRA_Symbolic_Reasoning.py
```

For other leading LLMs (GPT series/Llama2 family)
```sh
cd src
python Inference_in_context_learning.py
```

For evaluation
```sh
cd src
python Evaluation.py
```


## Datasets

All the datasets can be found [here](https://huggingface.co/datasets/sxiong/TGQA).

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
