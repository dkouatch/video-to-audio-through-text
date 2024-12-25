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

# Using eva_clip, .npy files
vgg_clip_visual_path = "/pscratch/sd/x/xiuliu/vggsound_clip_5fps/"

# vggsound audio_visual caption path
vgg_caption_path = "/global/homes/x/xiuliu/vggsound_audio_visual_caption.json"
vgg_test_wav_path = "/pscratch/sd/x/xiuliu/vggsound_audio_wav/test/"
test_split_set = set([fn[:-4] for fn in os.listdir(vgg_test_wav_path)])


with open(vgg_caption_path, "r") as fin:
    av_caption = json.load(fin)



v2a_instruction = []
for item in tqdm(av_caption):
    if item["file_id"] in test_split_set:
        split = "test"
    else:
        split = "train"
    v2a_instruction.append(
        {
            "input": "",
            "video_path": item["file_id"],
            "dataset": "vggsound",
            "output": item["audio_caption"].split("Audio caption: ")[-1],
            "split": split,
        }
    )

with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json", "w+") as fout:
    json.dump(v2a_instruction, fout, indent=2)