# this code is modified from lora_alpaca https://github.com/tloen/alpaca-lora under Apache-2.0 license
import os
from typing import List
import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset
import sys
import os, argparse
from tqdm import tqdm

import torchaudio
import json
import torch
from time import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import pandas as pd

vggsound_labels = pd.read_csv("/pscratch/sd/x/xiuliu/ltu/data/vggsound.csv", header=None).iloc[:, [0, -2]]
vggsound_labels = pd.Series(vggsound_labels.iloc[:, 1].values, index=vggsound_labels.iloc[:, 0]).to_dict()



# lora hyperparams
lora_r = 16
lora_alpha = 32
lora_dropout = 0.0
lora_target_modules= ["q_proj", "v_proj"]


# trick to load checkpoints correctly from HF
base_model = '/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/'
start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/v2a_lora_finetune_all_qformer/checkpoint-34000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/v2a_lora_finetune_all_llm_weights_vgg/checkpoint-34000/pytorch_model.bin'
# config = LlamaConfig.from_pretrained("/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/v2a_lora_finetune_all_llm_weights_vgg/config.json")
# model = LlamaForCausalLM(config)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference
cutoff_len = 108
config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)



# base_model = '/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/' # Vicuna-7B load first, then the fine-tuned differential weights
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/v2a_caption/checkpoint-30500/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/visual_finetune_vgg/checkpoint-5000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/visual_audio_finetune_vgg/checkpoint-22000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/finetune_v2a_scratch_all_vgg_audioset_ltu/checkpoint-134000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/visual_audio_finetune_full_vgg_audioset/checkpoint-117000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/visual_audio_finetune_vgg/checkpoint-22000/pytorch_model.bin'
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/v2a_lora_finetune_all/checkpoint-195000/pytorch_model.bin'
device_map = "auto"
state_dict = torch.load(start_model, map_location='cpu')
msg = model.load_state_dict(state_dict, strict=False)
print('load checkpoint', msg)
# for key, val in model.state_dict().items():
#     print(key)
#     if "bert" in key:
#         print(val)
#         print(val.requires_grad)
# for key, val in model.state_dict().items():
#     if "bert" in key:
#         print("HHH", val)
# exit(0)




# model = LlamaForCausalLM.from_pretrained(
#     base_model,
#     load_in_8bit=False,
#     torch_dtype=torch.float16,
#     device_map=device_map,
# )

# tokenizer = LlamaTokenizer.from_pretrained(base_model)

# tokenizer.pad_token_id = (
#     0  # unk. we want this to be different from the eos token
# )

# tokenizer.padding_side = "left"  # Allow batched inference
# cutoff_len = 108

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=108,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


# config = LoraConfig(
#     r=lora_r,
#     lora_alpha=lora_alpha,
#     target_modules=lora_target_modules,
#     lora_dropout=lora_dropout,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, config)



# msg = model.load_state_dict(state_dict, strict=False)
# print('load checkpoint', msg)
# exit(0)
model.cuda()

model.is_parallelizable = False
model.model_parallel = False

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()

instruction_templates = [
    # "Classify the audio event",
    # "Describe the visual scene.",
    # "Describe what you see and hear in the video."
    # "Based on what you see in the video, caption the sounds that match the video."
    # "Imagine possible sounds for this video, do not describe visual details.",
    "Describe possible sounds that could match the video.",
    # "What audio could be inferred from this video?",
    # "What sounds could match the video?",
    # "Infer the sounds that match the video.",
    # "What audio could best match this video?",
    # "What sound events could the video yields?",
    # "Caption possible sound events that describe the video.",
    # "What sound events make sense to this video?",
    # "Imagine audio events that match this video.",
]

# instruction = 'Close-ended question: Write an audio caption describing the sound.'
# temp, top_p, top_k = 0.1, 0.95, 500
temp, top_p, top_k = 0.1, 0.95, 500 
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temp,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=1.1,
    max_new_tokens=400,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    num_return_sequences=1
)

begin_time = time()
video_path_base = "/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json"
with open(video_path_base, "r") as fin:
    file_dict = json.load(fin)
test_file_dict = []
for item in file_dict:
    if item["split"] == "test":
        test_file_dict.append(item)
device = "cuda"
prompt_template = "alpaca_short" # The prompt template to use, will default to alpaca.
prompter = Prompter(prompt_template)
prompt = prompter.generate_prompt(instruction_templates[0], None)
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
# sel_video_set = ["0glBUluF4Yk_000175", "1PAb2MoavDc_000210", "3uuyQ4O0L68_000030", "5egGjbK3hSc_000026",
#                   "7fft0a682rE_000000", "Br2gA432RWo_000182", 'E4duB2A-ces_000050', "E7qRkUYu580_000307",
#                 "_hd1tNgkZz8_000009", "HOhpKjQWqxc_000030", "h_Q_2UKcQX0_000368", "hRc5mInvrYA_000003",
#                 "_jB-IM_77lI_000000", "mwoQMUhI9nY_000139", "o8_9STVfnvI_000059", "PuvTDLP8lwM_000030",
#                 "TJ4aeESZ6e8_000100", "v44s14ocMV4_000033", "wfRWvI16wxw_000047", "x_5t0os3A2I_000009",
#                 "yKZ1wVwmhus_000283", "0f0Rq7e5yX8_000160", "VN5W9piaNOw_000063", "2PHV2xNjGVU_000126"]
video_base_path = "/pscratch/sd/x/xiuliu/vggsound_clip_5fps/"
pred_captions = {}
for video_item in tqdm(test_file_dict):
    video_name = video_item["video_path"]
    # video_name = "-fAGzY9rnaA_000030"#"LDoXsip0BEQ_000177" #"hRc5mInvrYA_000003" #"Bd-1gr7807g_000016" # ##"hRc5mInvrYA_000003"#"3xDZU7GME7E_000170" #"-fAGzY9rnaA_000030"# #"j1kM-hC44Ok_000002" # # #"sqGwflGZk4Y_000080" #"-fAGzY9rnaA_000030"# # # # "0f0Rq7e5yX8_000160"
    video_path = os.path.join(video_base_path, video_name + ".npy")
    gt_sentence = video_item["output"]

# for video_name in tqdm(sel_video_set):
    try:
        cur_video_input = torch.Tensor(np.load(video_path)).unsqueeze(0)
        # cur_video_input = torch.Tensor(np.load(os.path.join(video_base_path, video_name + ".npy"))).unsqueeze(0)
        # if torch.cuda.is_available() == False:
        #     pass
        # else:
        cur_video_input = cur_video_input.to(device).half()
        # print("HHH", cur_video_input.shape)

        # Without streaming
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generation_output = model.generate(
                    input_ids=input_ids.to(device),
                    video_input=cur_video_input,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=400,
                )
        # for s in generation_output.sequences:
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)[6:-4]
        pred = output[len(prompt):]
        pred_captions[video_name] = pred
        # print(video_name, "   ", pred)
        # print("gt_sentence: ", gt_sentence)
        # print("x"*10)
    except Exception as e:
        print(e)
        print("Problem with video: ", video_name)
        continue
with open("/pscratch/sd/x/xiuliu/ltu/data/video_llama_vggsound_gen_audio_captions.json", "w+") as fout:
    json.dump(pred_captions, fout, indent=2)
    # end_time = time()
    # pred = output[len(prompt):]
    # print(video_name, vggsound_labels[video_name[:-7]])
    # print(pred)
    # print(gt_sentence)
    # print("*"*30)