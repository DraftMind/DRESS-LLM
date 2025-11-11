#!/bin/bash

# python get_activations.py --dataset_name Shakespeare --model_dir /data1/LLM_models/Qwen1.5-14B-Chat --session_path Qwen1.5_Shakes
# python edit_weight.py --model_dir /data1/LLM_models/Qwen1.5-14B-Chat --session_path Qwen1.5_Shakes --num_heads 64 --alpha 3
# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 20
# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 50

python generate.py --model_dir ./Qwen1.5_Shakes/edited_model/seed_42_top_64_heads_alpha_3.0 --input_dataset dataset/mage.json --session_path Qwen1.5_Shakes --output_path result/mage_result.json