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
from transformers import GenerationConfig, GemmaForCausalLM, GemmaTokenizer, GemmaConfig
from utils.prompter import Prompter
import pandas as pd

vggsound_labels = pd.read_csv("/pscratch/sd/x/xiuliu/ltu/data/vggsound.csv", header=None).iloc[:, [0, -2]]
vggsound_labels = pd.Series(vggsound_labels.iloc[:, 1].values, index=vggsound_labels.iloc[:, 0]).to_dict()

# lora hyperparams
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules= ["q_proj", "v_proj"]


# trick to load checkpoints correctly from HF
base_model = '/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/gemma_2b/' # Vicuna-7B load first, then the fine-tuned differential weights
# start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/pretrain_v2a_scratch_all_vgg_audioset_ltu_gemma/checkpoint-65000/pytorch_model.bin'
start_model = '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/pretrain_all_visual_audio_mix_gemma/checkpoint-231000/pytorch_model.bin'
device_map = "auto"

model = GemmaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map=device_map,
)

tokenizer = GemmaTokenizer.from_pretrained(base_model)


tokenizer.padding_side = "left"  # Allow batched inference
cutoff_len = 108

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


config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


state_dict = torch.load(start_model, map_location='cpu')
for key, val in state_dict.items():
    print(key)
msg = model.load_state_dict(state_dict, strict=False)
print('load checkpoint', msg)


model.is_parallelizable = True
model.model_parallel = True

model.eval()
instruction_templates = [
    "Imagine possible sounds for this video.",
    "Describe possible sounds that could match the video.",
    "What audio could be inferred from this video?",
    "What sounds could match the video?",
    "Infer the sounds that match the video.",
    "What audio could best match this video?",
    "What sound events could the video yields?",
    "Caption possible sound events that describe the video.",
    "What sound events make sense to this video?",
    "Imagine audio events that match this video.",
]

# instruction = 'Close-ended question: Write an audio caption describing the sound.'
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
prompt_template = "gemma" # The prompt template to use, will default to alpaca.
prompter = Prompter(prompt_template)
prompt = prompter.generate_prompt(instruction_templates[0], None)
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

video_base_path = "/pscratch/sd/x/xiuliu/vggsound_clip_5fps/"
pred_captions = {}
for video_item in tqdm(test_file_dict):
    try:
        video_name = video_item["video_path"]
        video_path = os.path.join(video_base_path, video_name + ".npy")
        gt_sentence = video_item["output"]
        cur_video_input = torch.Tensor(np.load(video_path)).unsqueeze(0)
        if torch.cuda.is_available() == False:
            pass
        else:
            cur_video_input = cur_video_input.half().to(device)
        # print("HHH", cur_video_input.shape)

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids.to(device),
                video_input=cur_video_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)#[6:-4]
        end_time = time()
        pred = output.split("<start_of_turn>model")[-1]
        pred_captions[video_name] = pred
    except Exception as e:
        print("Problem with video {}".format(video_name))
        continue
with open("/pscratch/sd/x/xiuliu/ltu/data/gemma_vggsound_gen_audio_captions.json", "w+") as fout:
    json.dump(pred_captions, fout, indent=2)
    # print(video_name)
    # print(pred) 
    # print(gt_sentence)
    # print("*"*30)
