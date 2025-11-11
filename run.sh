#!/bin/bash

# python get_activations.py --dataset_name Shakespeare --model_dir ~/models/Qwen3-8B --session_path Qwen3_Shakes
python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 3
# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 20
# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 50