# python get_activations.py --dataset_name Shakespeare --model_dir "/PretainedModels/Qwen3-8B" --session_path "/your/session/Qwen3-8B_Shakespeare"
import os
import torch
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen_DRC, tokenized_tqa_gen_Shakespeare
import llama
import qwen2
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Daiyu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--session_path", type=str, default=None, help='session path')
    args = parser.parse_args()

    MODEL = args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    if args.dataset_name == "DRC": 
        with open("dataset/Train_DRC.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_DRC
    elif args.dataset_name == "Shakespeare": 
        with open("dataset/Train_Shakespeare.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_Shakespeare
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    print(len(dataset))
    prompts, labels = formatter(dataset, tokenizer)
    print(len(prompts), len(labels))

    
    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    import gc
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        layer_wise_activations_wanted = layer_wise_activations[:,-1,:].copy()
        head_wise_activations_wanted = head_wise_activations[:,-1,:].copy()
        del layer_wise_activations, head_wise_activations, _
        all_layer_wise_activations.append(layer_wise_activations_wanted)
        all_head_wise_activations.append(head_wise_activations_wanted)

        gc.collect()

    print("Saving labels")
    if not os.path.exists(f'{args.session_path}/features'):
        os.makedirs(f'{args.session_path}/features')
    
    np.save(f'{args.session_path}/features/labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'{args.session_path}/features/layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'{args.session_path}/features/head_wise.npy', all_head_wise_activations)
    

if __name__ == '__main__':
    main()