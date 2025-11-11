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
parser.add_argument('--session_path', type=str)
args = parser.parse_args()

a = 'top_'
b = '_heads_alpha_'
index_a = args.model_dir.find(a)
index_b = args.model_dir.find(b)
K = args.model_dir[index_a + len(a): index_b]
alpha = args.model_dir[index_b + len(b): ]
dump_path = K + '_' + alpha
print(dump_path)

# 从预训练模型加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

# 准备问题
questions = []
with open(args.input_dataset, 'r', encoding='utf-8') as file:
    data_list = json.load(file)
for QA in data_list:
    questions.append(QA["question"])

answers = []

# 读取基于数据集计算并保存好的转向向量
np.load.__defaults__=(None, True, True, 'ASCII')
probes = np.load(f"{args.session_path}/features/probes_" + dump_path + ".npy")
top_heads = np.load(f"{args.session_path}/features/top_heads_" + dump_path + ".npy")
np.load.__defaults__=(None, False, True, 'ASCII')
with open(f"{args.session_path}/features/activations_" + dump_path + ".pkl", 'rb') as f:
    activations_dict = pickle.load(f)
num_heads = model.config.num_attention_heads
activations = np.load(f"{args.session_path}/features/head_wise.npy")
activations = rearrange(activations, 'b l (h d) -> b l h d', h = num_heads)

svd_s_dict = {}
svd_Vh_dict = {}

def svd_decomposition(layer_no, head_no, X):
    from scipy.linalg import svd
    U, s, Vh = svd(X, full_matrices=False)
    '''
    X: (4096, 128), 4096个正负样本对之差
    U: (4089, 128)
    s: (128, ), sigma矩阵的主对角线元素(奇异值降序)
    Vh: (128, 128)
    '''
    # 保存s, Vh
    key = 'L' + str(layer_no) + 'H' + str(head_no)
    svd_s_dict[key] = s
    svd_Vh_dict[key] = Vh


def get_steering_vector(layer_no, head_no, vector, cur_activations):
    key = 'L' + str(layer_no) + 'H' + str(head_no)
    K = 64
    s = svd_s_dict[key]
    Vh = svd_Vh_dict[key]
    Vh = Vh[:K, :]
    x = vector
    V = Vh.T
    w = np.dot(Vh, x.T)
    w2 = np.dot(Vh, cur_activations.T)
    head_activations = activations[:,layer_no,head_no,:]
    correct_activations = head_activations[::2, :]
    correct_activations = np.mean(correct_activations, axis=0)
    w4 = np.dot(Vh, correct_activations.T)
    w *= (1.0 + 0.5 * np.sign(w) * (w4 - w2))
    xx = np.dot(V, w)
    return xx


def get_activations(question):
    # prompt = tokenizer_1(question, return_tensors = 'pt').input_ids
    # 在不同的layer, head上计算question的activation与probe的相似度
    # bias=0时计算activation
    with torch.no_grad():
        for layer_no, heads in activations_dict.items():
            displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
            device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
            displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
            bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
            model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    prompt = question
    all_head_wise_activations = []
    device = "cuda"
    layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
    all_head_wise_activations.append(head_wise_activations[:,-1,:])
    head_wise_activations = rearrange(all_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    weights = []
    with torch.no_grad():
        for layer_no, heads in activations_dict.items():
            displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
            for head_no, vector in activations_dict[layer_no].items():
                cur_activations = head_wise_activations[:,layer_no,head_no,:].flatten()

                s_vector = get_steering_vector(layer_no, head_no, vector, cur_activations)
                displacement[head_no] = s_vector
                
            device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
            displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
            bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
            model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
    return



def my_generate(w0, q_tokens, inputs):
    generated = inputs["input_ids"]
    sequence = []
    max_length = 600
    layer_num = 40
    avg_weights = [w0]
    for i in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to('cuda:0')
            generated = torch.cat((generated, token), dim=1)
            q_tokens = torch.cat((q_tokens, token), dim=1)
            sequence.append(token.cpu().numpy()[0][0])
            # get_activations(q_tokens)

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: 
                break
    return sequence


print("为所有head上的转向向量作svd分解并保存")
for layer_no, heads in activations_dict.items():
        for head_no, vector in activations_dict[layer_no].items():
            head_activations = activations[:,layer_no,head_no,:]
            correct_activations = head_activations[::2, :]
            incorrect_activations = head_activations[1::2, :]
            correct_activations = correct_activations - incorrect_activations
            svd_decomposition(layer_no, head_no, correct_activations)
print("分解完毕")


for index, question in enumerate(questions):

    q_tokens = tokenizer(question, return_tensors = 'pt').input_ids
    # w0 = get_activations(q_tokens)
    w0 = None
    
    question = format_with_chat_template(tokenizer, question=question)
    
    inputs = tokenizer(question, return_tensors='pt')
    
    #Normal generation
    # 调用model 对 inputs进行generate 并对模型生成内容进行解码
    outputs = model.generate(**{k: v.to(model.device) for k, v in inputs.items()}, max_length=600, num_return_sequences=1, top_k=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    
    # #DRESS generation
    # inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    # sequence = my_generate(w0, q_tokens.to('cuda:0'), inputs)
    # # print(sequence)
    # answer = tokenizer.decode(sequence, skip_special_tokens=False)
    print(index, answer)
    answers.append(answer)
    


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