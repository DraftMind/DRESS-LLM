#!/usr/bin/env python
# Usage:
# python generate.py \
#   --model_dir "/your/session/Qwen3-8B_Shakespeare/edited_model/seed_42_top_64_heads_alpha_3.0" \
#   --input_dataset "dataset/Valid_Shakespeare.json" \
#   --session_path "/your/session/Qwen3-8B_Shakespeare" \
#   --output_path "result.json"
import os
import torch
import numpy as np
import pickle
import sys
sys.path.append('../')
from utils import get_llama_activations_bau, format_with_chat_template
import argparse
import json
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--input_dataset', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--engine', type=str, choices=['hf', 'vllm'], default='hf', help='Inference engine: hf (Transformers) or vllm')
args = parser.parse_args()

# 从预训练模型加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = None
llm = None
sampling_params = None
if args.engine == 'hf':
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
else:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError("vLLM is not installed or failed to import. Please install vllm to use --engine vllm.") from e
    llm = LLM(model=args.model_dir, max_model_len=2048)
    sampling_params = SamplingParams(max_tokens=600, top_k=1)

# 准备问题
questions = []
with open(args.input_dataset, 'r', encoding='utf-8') as file:
    data_list = json.load(file)
for QA in data_list:
    questions.append(QA["question"])

answers = []

import random
questions = random.sample(questions, min(len(questions), 500))
prompts = []

for question in questions:
    prompt = question    
    prompt = format_with_chat_template(tokenizer, question=prompt)
    prompts.append(prompt)


if args.engine == 'hf':
    for index, question in enumerate(tqdm(prompts, desc="Generating answers")):        
        inputs = tokenizer(question, return_tensors='pt')
        outputs = model.generate(**{k: v.to(model.device) for k, v in inputs.items()}, max_length=600, num_return_sequences=1, top_k=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(answer)
        answers.append(answer)
elif args.engine == 'vllm':
    with tqdm(total=len(prompts), desc="vLLM Generating", unit="sample") as pbar:
        outputs = llm.generate(prompts, sampling_params)
        for out in outputs:
            answer = out.outputs[0].text if out.outputs else ""
            answers.append(answer)
            pbar.update(1)

output_data = []
for i in range(len(questions)):
    dict = {}
    dict["question"] = questions[i]
    dict["daiyu_answer"] = answers[i]
    dict["model_path"] = args.model_dir
    output_data.append(dict)
###########################
with open(args.output_path, 'w', encoding='utf-8') as new_file:
    json.dump(output_data, new_file, ensure_ascii=False, indent=4)