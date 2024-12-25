import os, json

all_audio_visual_caption_data = []

with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_v2a_instruction.json", "r") as fin:
    all_audio_visual_caption_data.extend(json.load(fin))

with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_visual_instruction.json", "r") as fin:
    all_audio_visual_caption_data.extend(json.load(fin))


with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_instruction.json", "w+") as fout:
    json.dump(all_audio_visual_caption_data, fout, indent=2)