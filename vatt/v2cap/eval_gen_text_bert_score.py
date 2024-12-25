import bert_score
from bert_score import score
import json
from glob import glob
import os
from tqdm import tqdm


# with open("/pscratch/sd/x/xiuliu/ltu/data/llama_vggsound_gen_audio_captions_full_audio_cap_only_model.json", "r") as fin: #gemma_vggsound_gen_audio_captions
#     cands_dict = json.load(fin)
# cands_dict = {}
# for fp in tqdm(glob("/pscratch/sd/x/xiuliu/video_llama_v2a_captions/*.txt")):
#     fn = os.path.basename(fp)[:-4]
#     caption = ""
#     with open(fp, "r") as fin:
#         for line in fin:
#             caption = line.strip()
#             break
#     cands_dict[fn] = caption


with open("/pscratch/sd/x/xiuliu/ltu/data/video_llama_vggsound_gen_audio_captions.json", "r") as fin: #gemma_vggsound_gen_audio_captions
    cands_dict = json.load(fin)


# with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_visual_instruction.json", "r") as fin:
#     cands_list = json.load(fin)
# cands_dict = dict()
# for item in cands_list:
#     if item["split"] == "test":
#         cands_dict[item["video_path"]] = item["output"]

video_path_base = "/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json"
with open(video_path_base, "r") as fin:
    file_dict = json.load(fin)
ref_dict = dict()
for item in file_dict:
    if item["split"] == "test":
        ref_dict[item["video_path"]] = item["output"]
cands, refs = [], []
for key, val in cands_dict.items():
    cands.append(val)
    refs.append(ref_dict[key])
    # print(val, ref_dict[key])

P, R, F1 = score(cands, refs, lang='en', verbose=True)
print(f"System level F1 score: {F1.mean():.3f}")
print(f"System level Precision score: {P.mean():.3f}")
print(f"System level Recall score: {R.mean():.3f}")