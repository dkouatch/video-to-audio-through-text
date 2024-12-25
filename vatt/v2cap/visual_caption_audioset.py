#  {
#   "instruction": "Identify the sounds in the recording? Output labeled acoustic traits.",
#   "input": "",
#   "audio_id": "/data/sls/scratch/yuangong/avbyol2/egs/vggsound/preprocess/data/audio_16k/BT7h7bpL0wE_000030.flac",
#   "dataset": "vggsound_train",
#   "task": "cla_label_des",
#   "output": "Labels with acoustic features: High-frequency and sharp -> Chicken clucking"
#  }
from tqdm import tqdm


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

import os, json
import numpy as np
import glob

# Using eva_clip, .npy files

split = "unbalanced_train"
# split = "balanced_train"
# split = "eval"
audioset_clip_visual_path = "/pscratch/sd/x/xiuliu/audioset_split_eva_clip_encodec_tokens_hdf5/{}/".format(split)

split_file_to_path = dict()
# for file_path in tqdm(glob.glob(audioset_clip_visual_path + "*")):
for file_path in tqdm(glob.glob(audioset_clip_visual_path + "*/*")):
    file_name = os.path.basename(file_path).split(".")[0]
    split_file_to_path[file_name] = file_path
    # print(file_name)


# vggsound audio_visual caption path
# audioset_caption_path = "/global/homes/x/xiuliu/audioset_{}_audio_caption_all.json".format(split)
audioset_caption_path = "/global/homes/x/xiuliu/audioset_{}_visual_caption_all.json".format(split)



with open(audioset_caption_path, "r") as fin:
    caption = json.load(fin)




v_instruction = []
for file_name, visual_cap in tqdm(caption.items()):
    if file_name not in split_file_to_path: 
        print(file_name)
        continue
    v_instruction.append(
        {
            "input": "",
            "video_path": split_file_to_path[file_name],
            "dataset": "audioset",
            "caption": "visual",
            "output": visual_cap,
            "split": "train" if "train" in split else "test",
        }
    )
# print("HHHH", len(v2a_instruction))
with open("/pscratch/sd/x/xiuliu/ltu/data/audioset_{}_visual_instruction.json".format(split), "w+") as fout:
    json.dump(v_instruction, fout, indent=2)