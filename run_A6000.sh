#!/bin/bash
# # 第一块：在GPU 0上运行（块内串行）
# {
#     # python get_activations.py --dataset_name Shakespeare --model_dir /data1/LLM_models/Qwen3-4B --session_path session/Qwen3_4B_Shakes
#     CUDA_VISIBLE_DEVICES=0 python edit_weight.py --model_dir /data1/LLM_models/Qwen3-4B --session_path session/Qwen3_4B_Shakes --num_heads 64 --alpha 1.0
#     CUDA_VISIBLE_DEVICES=0 python generate.py --model_dir session/Qwen3_4B_Shakes/edited_model/seed_42_top_64_heads_alpha_1.0 --input_dataset dataset/Valid_Shakespeare.json --output_path result/Shakespeare_Qwen3-4B_head_64_alpha_1.0.json --engine vllm
# } &

# # 第二块：在GPU 1上运行（块内串行）
# {
#     CUDA_VISIBLE_DEVICES=1 python get_activations.py --dataset_name Shakespeare --model_dir /data1/LLM_models/Qwen3-4B-Instruct-2507 --session_path session/Qwen3_4BInstruct_Shakes
#     CUDA_VISIBLE_DEVICES=1 python edit_weight.py --model_dir /data1/LLM_models/Qwen3-4B-Instruct-2507 --session_path session/Qwen3_4BInstruct_Shakes --num_heads 64 --alpha 1.0
#     CUDA_VISIBLE_DEVICES=1 python generate.py --model_dir session/Qwen3_4BInstruct_Shakes/edited_model/seed_42_top_64_heads_alpha_1.0 --input_dataset dataset/Valid_Shakespeare.json --output_path result/Shakespeare_Qwen3-4BInstruct_head_64_alpha_1.0.json --engine vllm
# } &

# 第三块：在GPU 2上运行（块内串行）
{
    CUDA_VISIBLE_DEVICES=2 python generate.py --model_dir session/Qwen3_14B_Shakes/edited_model/seed_42_top_64_heads_alpha_1.0 --input_dataset dataset/Valid_Shakespeare.json --output_path result/Shakespeare_Qwen3-14B_head_64_alpha_1.0_mygenerate.json --session_path session/Qwen3_14B_Shakes --engine mygenerate
} &

# 等待所有后台任务完成
wait