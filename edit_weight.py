# python edit_weight.py --model_dir "/PretainedModels/Qwen3-8B" --session_path "/your/session/Qwen3-8B_Shakespeare" --num_heads 64 --alpha 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import sys
sys.path.append('../')
from utils import get_interventions_dict, get_top_heads, get_separated_activations, zero_qwen3_attention_biases
import llama
import qwen2
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--session_path", type=str, default=None, help='session path')
    # 以上为必需参数
    parser.add_argument('--num_heads', type=int, default=96, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    # set seeds
    print("set seeds")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    print("create model")
    MODEL = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    num_zeroed = zero_qwen3_attention_biases(model)
    print(f"Zeroed {num_zeroed} attention biases")
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    print("load activations")
    head_wise_activations = np.load(f"{args.session_path}/features/head_wise.npy")
    labels = np.load(f"{args.session_path}/features/labels.npy")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    print(head_wise_activations.shape)
    
    dataset_len = head_wise_activations.shape[0] // 2

    # tuning dataset: no labels used, just to get std of activations along the direction
    tuning_activations = np.load(f"{args.session_path}/features/head_wise.npy")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"{args.session_path}/features/labels.npy")

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    train_idxs = np.arange(dataset_len)

    # pick a val set using numpy
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    # get directions
    com_directions = None
    # top_heads, probes do not depend on alpha, cache and reuse if possible
    probes_path = f"{args.session_path}/features/probes_{args.num_heads}.npy"
    top_heads_path = f"{args.session_path}/features/top_heads_{args.num_heads}.npy"
    if os.path.exists(probes_path) and os.path.exists(top_heads_path):
        print(f"Loading cached probes and top_heads from {probes_path} and {top_heads_path}")
        probes = np.load(probes_path, allow_pickle=True)
        top_heads = np.load(top_heads_path, allow_pickle=True)
    else:
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        np.save(probes_path, probes)
        np.save(top_heads_path, top_heads)

    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

    # Try to load cached per-head delta (correct - incorrect) to reuse across alphas
    base_delta_path = f"{args.session_path}/features/displacement_base_{args.num_heads}.pkl"
    base_deltas = None
    if os.path.exists(base_delta_path):
        try:
            with open(base_delta_path, 'rb') as f:
                base_deltas = pickle.load(f)
            print(f"Loaded cached displacement base from {base_delta_path}")
        except Exception as e:
            print(f"Failed to load cached displacement base ({e}), will recompute.")
            base_deltas = None

    # If no cache available, we'll populate and save it after computing
    base_deltas_to_save = {} if base_deltas is None else None

    activations_dict = {} # save alpha-scaled displacement for record
    for head_out_name, list_int_vec in tqdm(interventions.items()):
        layer_no = int(head_out_name.split('.')[2])
        # Use the true per-head dimension from activations to size displacement correctly
        head_dim = int(tuning_activations.shape[-1])
        displacement = np.zeros((int(num_heads), head_dim))
        activations_dict[layer_no] = {} # save
        for head_no, head_vec, std in list_int_vec:

            # Use cached delta if available; otherwise compute and cache
            cached_delta = None
            if base_deltas is not None:
                if layer_no in base_deltas and head_no in base_deltas[layer_no]:
                    cached_delta = base_deltas[layer_no][head_no]

            if cached_delta is None:
                activations = tuning_activations[:,layer_no,head_no,:]
                correct_activations = activations[::2, :]
                incorrect_activations = activations[1::2, :]
                correct_activations = np.mean(correct_activations, axis=0)
                incorrect_activations = np.mean(incorrect_activations, axis=0)
                delta_vec = (correct_activations - incorrect_activations)
                # prepare for saving if needed
                if base_deltas_to_save is not None:
                    if layer_no not in base_deltas_to_save:
                        base_deltas_to_save[layer_no] = {}
                    base_deltas_to_save[layer_no][head_no] = delta_vec
            else:
                delta_vec = cached_delta

            displacement[head_no] = args.alpha * delta_vec
            
            activations_dict[layer_no][head_no] = displacement[head_no] # save
      
        device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    # Save base deltas for future fast re-use if we computed them this run
    if base_deltas_to_save is not None:
        with open(base_delta_path, 'wb') as f:
            pickle.dump(base_deltas_to_save, f)
        print(f"Saved displacement base (delta vectors) to {base_delta_path}")

    with open(f"{args.session_path}/features/activations_{args.num_heads}_{args.alpha:.1f}.pkl", 'wb') as f:
        pickle.dump(activations_dict, f)

    print("save results")
    if not os.path.exists(f"{args.session_path}/edited_model"):
        os.makedirs(f"{args.session_path}/edited_model")
    save_folder = f"{args.session_path}/edited_model/seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha:.1f}"
    if os.path.exists(save_folder):
      shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    model.config.attention_bias = True
    model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
    tokenizer.save_pretrained(save_folder)


if __name__ == "__main__":
    main()
