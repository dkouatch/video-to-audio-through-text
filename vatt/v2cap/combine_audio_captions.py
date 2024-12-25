import os, json

# all_audio_caption_data = []
# for split in ["unbalanced_train", "balanced_train", "eval"]:
#     with open("/pscratch/sd/x/xiuliu/ltu/data/audioset_{}_v2a_instruction.json".format(split), "r") as fin:
#         all_audio_caption_data.extend(json.load(fin))

# with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json", "r") as fin:
#     all_audio_caption_data.extend(json.load(fin))


with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_v2a_instruction.json", "r") as fin:
    all_audio_caption_data = json.load(fin)

with open("/pscratch/sd/x/xiuliu/ltu/data/ltu_adapt_v2a_instruction.json", "r") as fin:
    all_audio_caption_data.extend(json.load(fin))

with open("/pscratch/sd/x/xiuliu/ltu/data/dataset_all_v2a_plus_ltu_instruction.json", "w+") as fout:
    json.dump(all_audio_caption_data, fout, indent=2)

print(len(all_audio_caption_data))