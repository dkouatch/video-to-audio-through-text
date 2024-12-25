import os, json

all_files = []
with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json", "r") as fin:
    all_files.extend(json.load(fin))
with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_visual_instruction.json", "r") as fin:
    all_files.extend(json.load(fin))

with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_all_instruction.json", "w+") as fout:
    json.dump(all_files, fout, indent=2)