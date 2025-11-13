# DRESS-LLM
Code and Benchmark Dataset for paper DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing.

**[Update 25/2/2] 🔥 Our work has been accepted to ICLR'25!**

**Paper Link** 🔗: https://openreview.net/forum?id=mNVR9jJYqK (ICLR'25) / https://arxiv.org/abs/2501.14371 (arXiv) 


## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Workflow](#workflow)


## Installation
Run the following commands to set things up. (baukit 可以先clone https://github.com/davidbau/baukit + 手动安装，下面方法容易报连接错误)
```
git clone XXXX (This Github Link)
cd DRESS-LLM
conda env create -f environment.yml 
conda activate DRESSllm
```

## Dataset

`dataset/Train_Shakespeare.json` and `dataset/Train_DRC.json` are the two training datasets we created for the language style transfer task, namely the Shakespearean style and the dialogue style of characters in Dream of the Red Chamber. For each piece of data,  `question`, `correct_answers`, and `incorrect_answers` are respectively input, target style output, and ordinary style output (i.e., generic featureless output), and the two outputs are almost semantically equivalent. 

The testing sets are `dataset/Valid_Shakespeare.json` and `dataset/Valid_DRC.json` respectively. We only use the `question` for testing.

For more information about the dataset, please refer to the paper.

## For Qwen3
1. At line 183 of file lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py. In the init function of forward add the following code 
```
self.head_out = nn.Identity()
```
Within forward function of forward, add follows
```
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.head_out(attn_output)  # New Code
        attn_output = self.o_proj(attn_output)
```

2. Change the attention_bias config of Qwen3 Model to be True

3. **If using vLLM for inference**: Fix the vLLM Qwen3 implementation bug
   - File: `lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py`
   - Line 110 (in `Qwen3Attention.__init__`): Change `bias=False` to `bias=qkv_bias`
   - This ensures vLLM respects the `attention_bias` config when loading o_proj weights
   ```python
   # Original (BUG):
   self.o_proj = RowParallelLinear(..., bias=False, ...)
   
   # Fixed:
   self.o_proj = RowParallelLinear(..., bias=qkv_bias, ...)
   ```

## Workflow

(1) Get activations:
```
python get_activations.py \
  --dataset_name Shakespeare \
  --model_dir "/PretainedModels/Qwen3-8B" \
  --session_path "/your/session/Qwen3-8B_Shakespeare"
```
You need to fill in the dataset name (`DRC` or `Shakespeare`), the model path, and a `session_path` where outputs are stored. Outputs are saved under `{session_path}/features`.

(2) Edit and save the model:
```
python edit_weight.py \
  --model_dir "/PretainedModels/Qwen3-8B" \
  --session_path "/your/session/Qwen3-8B_Shakespeare" \
  --num_heads 64 \
  --alpha 3
```
This reads features from `{session_path}/features` and saves the edited model to `{session_path}/edited_model/seed_42_top_64_heads_alpha_3.0`.

(3) Generate answers on the test set:
```
python generate.py \
  --model_dir "/your/session/Qwen3-8B_Shakespeare/edited_model/seed_42_top_64_heads_alpha_3.0" \
  --input_dataset "dataset/Valid_Shakespeare.json" \
  --session_path "/your/session/Qwen3-8B_Shakespeare" \
  --output_path "result.json"
```
The reasoning adopts the [DRESSing UP LLM] strategy, adaptively adjusting the steering intensity in the style subspace to achieve higher generation quality. Results are saved in `result.json`.
Results will be saved in `result.json`.

---

**Finally, if you find our work helpful or interesting, please don't forget to cite us!**

ICLR Version
```
@inproceedings{
ma2025dressing,
title={{DRESS}ing Up {LLM}: Efficient Stylized Question-Answering via Style Subspace Editing},
author={Xinyu Ma and Yifeng Xu and Yang Lin and Tianlong Wang and Xu Chu and Xin Gao and Junfeng Zhao and Yasha Wang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=mNVR9jJYqK}
}
```
arXiv Version
```
@misc{ma2025dressingllm,
      title={DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing}, 
      author={Xinyu Ma and Yifeng Xu and Yang Lin and Tianlong Wang and Xu Chu and Xin Gao and Junfeng Zhao and Yasha Wang},
      year={2025},
      eprint={2501.14371},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2501.14371}, 
}
```
