import os, json

all_visual_caption_data = []
for split in ["unbalanced_train"]:
    with open("/pscratch/sd/x/xiuliu/ltu/data/audioset_{}_visual_instruction.json".format(split), "r") as fin:
        all_visual_caption_data.extend(json.load(fin))

with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_visual_instruction.json", "r") as fin:
    all_visual_caption_data.extend(json.load(fin))


with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_visual_instruction.json", "w+") as fout:
    json.dump(all_visual_caption_data, fout, indent=2)
