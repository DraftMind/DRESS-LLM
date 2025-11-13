#!/bin/bash

# python get_activations.py --dataset_name Shakespeare --model_dir ~/models/Qwen3-4B --session_path session/Qwen3_4B_Shakes
python edit_weight.py --model_dir ~/models/Qwen3-4B --session_path session/Qwen3_4B_Shakes --num_heads 16 --alpha 1.0
python edit_weight.py --model_dir ~/models/Qwen3-4B --session_path session/Qwen3_4B_Shakes --num_heads 16 --alpha 3.0

# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 20
# python edit_weight.py --model_dir ~/models/Qwen3-8B --session_path ./Qwen3_Shakes --num_heads 64 --alpha 50
# python generate.py --model_dir session/Qwen3_4B_Shakes/edited_model/seed_42_top_64_heads_alpha_3.0 --input_dataset dataset/Valid_Shakespeare.json --output_path result/Shakespeare_4B.json --engine hf