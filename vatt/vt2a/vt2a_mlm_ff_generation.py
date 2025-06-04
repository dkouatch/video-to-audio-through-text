import h5py
from os.path import join as pjoin
import argparse, os
import torch
from tqdm import tqdm
import numpy as np
from vt2a.util import instantiate_from_config
from omegaconf import OmegaConf
import io
import codecs as cs
import random
import time
from transformers import LlamaTokenizer
from vt2a.data.prompter import Prompter


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

def tokenize(tokenizer, prompt, add_eos_token=True, max_length=64):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    return result


def load_model_from_config(config, ckpt, verbose=True, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    keys = list(sd.keys())
    model = instantiate_from_config(config.model)
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    print(len(m), len(u))
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="The path to save model output (audio tokens)",
        default='/path/to/save'
    )

    parser.add_argument(
        "--img_embs_dir",
        type=str,
        required=False,
        help="The path to ib data dir",
        default='/path/to/vggsound_clip_5fps/'
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        required=False,
        help="The path to vggsound_test.txt dir",
        default="./",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        default='./configs/vt2a_mlm_alibi_mix_large_uni_encodec_full_stage_2.yaml'
        # default='./configs/vt2a_mlm_alibi_mix_large_uni_encodec_vgg.yaml'
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        # default='/path/to/vatt_llama.ckpt'
        # default='/path/to/vatt_llama_T.ckpt'
    )
    args = parser.parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    img_embs_dir = args.img_embs_dir
    meta_dir = args.meta_dir
    ckpt_path = args.ckpt_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(args.config_file)
    model = load_model_from_config(config, ckpt_path)
    model = model.to(device)
    all_time = []

    prompter = Prompter("alpaca_short")
    tokenizer = LlamaTokenizer.from_pretrained("/path/to/pretrained_mdls/vicuna/")
    tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"

    import json
    with open("/path/to/vgg_test_5_per_class_for_retrieval_cleaned.json", "r") as fin:
        test_small_split = json.load(fin)["data"]
    test_small_set = set()
    for item in test_small_split:
        test_small_set.add(item["video_id"])

    # GT captions
    VGG_PROMPT_PATH = '/path/to/vggsound_v2a_instruction.json'
    with open(VGG_PROMPT_PATH, "r") as fin:
        instructions = json.load(fin)
    text_prompts = dict()
    for item in instructions:
        text_prompts[item["video_path"]] = item["output"]

    # Llama generated audio captions
    # VGG_PROMPT_PATH = "/path/to/llama_vggsound_gen_audio_captions_full_audio_cap_only_model.json"
    # with open(VGG_PROMPT_PATH, "r") as fin:
    #     text_prompts = json.load(fin)


    # VGG_PROMPT_PATH = "/path/to/gemma_vggsound_gen_audio_captions.json"
    # with open(VGG_PROMPT_PATH, "r") as fin:
    #     text_prompts = json.load(fin)
    STAGE = 2

    cnt = 0
    with cs.open(pjoin(meta_dir, f'vggsound_test.txt'), "r") as f:
        for line in tqdm(f.readlines()):
            if cnt >= 15546 // 2:
                continue
            cnt += 1
            fn = line.strip()
            if os.path.exists(pjoin(save_path, "{}.npy".format(fn))):
                continue
            
            img_embs_path = pjoin(img_embs_dir, fn + ".npy")
            img_embs = np.load(img_embs_path)
            img_embs = torch.from_numpy(img_embs).float().unsqueeze(0)

            if STAGE == 1:
                text_ix = 2 #random.randint(0, len(VGG_STAGE_1_TEMPLATES) - 1)
                select_prompt = VGG_STAGE_1_TEMPLATES[text_ix]
            else:
                select_prompt = text_prompts[fn]
            full_prompt = prompter.generate_prompt(select_prompt)
            tokenized_full_prompt = tokenize(tokenizer, full_prompt)
            input_ids = torch.Tensor(tokenized_full_prompt["input_ids"]).long().unsqueeze(0)
            attention_mask = torch.Tensor(tokenized_full_prompt["attention_mask"]).float().unsqueeze(0)

            video_att_mask = torch.ones([1, 10]) # batch size [2, 10]
            attention_mask = torch.concat([video_att_mask, attention_mask], dim=1)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    start = time.time()
                    x = {
                        "input_ids": input_ids.to(device),
                        "attention_mask": attention_mask.half().to(device),
                        "video_input": img_embs.half().to(device)
                    }
                    out = model.generate_audio(x, cfg_coef=5, mask_temperature=25.5) #temp 12.5 15.5
                    end = time.time()
                    all_time.append(end - start)
                    save_fn = "{}.npy".format(fn)
                    pred_save_dac_path = pjoin(save_path, save_fn)
                    assert out.shape == (1, 4, 500)
                    np.save(pred_save_dac_path, out[0].cpu().numpy())

    print(np.mean(all_time))
