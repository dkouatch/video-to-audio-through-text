import h5py
from os.path import join as pjoin
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import io
from tqdm import tqdm
import torch
import codecs as cs
import json
import random
import os
import glob
from transformers import LlamaTokenizer, GemmaTokenizer
from vt2a.data.prompter import Prompter


BASE_PATH = "/pscratch/sd/x/xiuliu/audioset_split_eva_clip_encodec_tokens_hdf5/"
VGG_AUDIO_TOKENS_PATH = '/pscratch/sd/x/xiuliu/meta_pretrain_vgg_encodec_tokens'
VGG_IMG_EMBS_PATH = '/pscratch/sd/x/xiuliu/vggsound_clip_5fps'
VGG_PROMPT_PATH = '/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json'

META_DIR = '/pscratch/sd/x/xiuliu'

VGG_STAGE_1_TEMPLATES = [
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


def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

class MixDataset(Dataset):
    def __init__(self, split, stage=1, clip_interp=True, prompt_template_name="alpaca_short", base_model="/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/"):
        self.split = split
        self.clip_interp = clip_interp
        if prompt_template_name == "alpaca_short":
            self.prompter = Prompter(prompt_template_name)
            self.tokenizer = LlamaTokenizer.from_pretrained(base_model)

            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
        else:
            self.prompter = Prompter(prompt_template_name)
            self.tokenizer = GemmaTokenizer.from_pretrained(base_model)

        self.tokenizer.padding_side = "left"
        self.data = []

        with cs.open(pjoin(META_DIR, f'vggsound_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.data.append((line.strip(), 'vgg'))
        
        self.stage = stage
        if stage == 1:
            self.text_prompts = VGG_STAGE_1_TEMPLATES
        else:
            self.text_prompts = dict()
            with open(VGG_PROMPT_PATH, "r") as fin:
                instructions = json.load(fin)
            for item in instructions:
                self.text_prompts[item["video_path"]] = item["output"]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, data_n =  self.data[idx]
        audio_tokens = np.load(pjoin(VGG_AUDIO_TOKENS_PATH, fn + ".npy"))
        audio_tokens = torch.from_numpy(audio_tokens).permute(1, 0)
        clip_emb = np.load(pjoin(VGG_IMG_EMBS_PATH, fn + ".npy"))
        clip_emb = torch.from_numpy(clip_emb).float()
        no_help = False
        if self.stage == 1 or (self.stage == 2 and fn not in self.text_prompts):
           text_ix = random.randint(0, len(VGG_STAGE_1_TEMPLATES) - 1)
           select_prompt = VGG_STAGE_1_TEMPLATES[text_ix]
           no_help = True
        else:
            select_prompt = self.text_prompts[fn]

        full_prompt = self.prompter.generate_prompt(select_prompt)
        tokenized_full_prompt = self.tokenize(full_prompt)
        input_ids = tokenized_full_prompt["input_ids"]
        attention_mask = tokenized_full_prompt["attention_mask"]

        

        if self.split == "train":
            aug_prob = random.random()
            if aug_prob < 0.5 and no_help:
                aug_idx = random.randint(0, len(self.data)-1)
                aug_fn, aug_data_n =  self.data[aug_idx]
                aug_audio_tokens = np.load(pjoin(VGG_AUDIO_TOKENS_PATH, aug_fn + ".npy"))
                aug_audio_tokens = torch.from_numpy(aug_audio_tokens).permute(1, 0)
                aug_clip_emb = np.load(pjoin(VGG_IMG_EMBS_PATH, aug_fn + ".npy"))
                aug_clip_emb = torch.from_numpy(aug_clip_emb).float()
            
                split_pos = random.randint(1, 8)
                audio_tokens = torch.cat([audio_tokens[:, :split_pos*50], aug_audio_tokens[:, split_pos*50:]], dim=1)           
                clip_emb = torch.cat([clip_emb[:int(split_pos*5)], aug_clip_emb[int(split_pos*5):]], dim=0)
            elif 0.5 < aug_prob < 0.6:
                roll_idx = np.random.randint(-clip_emb.shape[0], clip_emb.shape[0])
                clip_emb = torch.roll(clip_emb, roll_idx, 0)
                audio_tokens = torch.roll(audio_tokens, roll_idx*10, 1)



        return {
            'audio_tokens': audio_tokens,
            'video_inputs': clip_emb,
            'input_ids': input_ids,
            "attention_mask": torch.Tensor(attention_mask)
        }
    

    def tokenize(self, prompt, add_eos_token=True, max_length=64):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result