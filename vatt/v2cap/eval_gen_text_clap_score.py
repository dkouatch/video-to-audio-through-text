import numpy as np
import librosa
import torch
import laion_clap
import glob, json, os
from tqdm import tqdm

device = torch.device('cuda:0') 

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt() # download the default pretrained checkpoint.


# Directly get audio embeddings from audio files, but return torch tensor
audio_file_path = "/pscratch/sd/x/xiuliu/vggsound_audio_wav/test/*.wav"
audio_file = glob.glob(audio_file_path)

text_data = []

# GT Captions
# video_path_base = "/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json"
# with open(video_path_base, "r") as fin:
#     file_dict = json.load(fin)
# ref_dict = dict()
# for item in file_dict:
#     if item["split"] == "test":
#         ref_dict[item["video_path"]] = item["output"]
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(ref_dict[fn])

# Llava Captions

# video_path_base = "/pscratch/sd/x/xiuliu/ltu/data/vggsound_visual_instruction.json"
# with open(video_path_base, "r") as fin:
#     file_dict = json.load(fin)
# ref_dict = dict()
# for item in file_dict:
#     if item["split"] == "test":
#         ref_dict[item["video_path"]] = item["output"]
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(ref_dict[fn])

# Gemma generated captions:
# with open("/pscratch/sd/x/xiuliu/ltu/data/gemma_vggsound_gen_audio_captions.json", "r") as fin: #
#     ref_dict = json.load(fin)
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(ref_dict[fn])

# Llama generated captions:
# with open("/pscratch/sd/x/xiuliu/ltu/data/llama_vggsound_gen_audio_captions_full_audio_cap_only_model.json", "r") as fin: #
#     ref_dict = json.load(fin)
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(ref_dict[fn])

# Video-LLAMA zero-shot
# cands_dict = {}
# for fp in tqdm(glob.glob("/pscratch/sd/x/xiuliu/video_llama_v2a_captions/*.txt")):
#     fn = os.path.basename(fp)[:-4]
#     caption = ""
#     with open(fp, "r") as fin:
#         for line in fin:
#             caption = line.strip()
#             break
#     cands_dict[fn] = caption
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(cands_dict[fn])

# with open("/pscratch/sd/x/xiuliu/ltu/data/vggsound_v2a_instruction.json", "r") as fin: #
#     ref_dict = json.load(fin)
# for fp in audio_file:
#     fn = os.path.basename(fp)[:-4]
#     text_data.append(ref_dict[fn])


with open("/pscratch/sd/x/xiuliu/ltu/data/video_llama_vggsound_gen_audio_captions.json", "r") as fin: #
    ref_dict = json.load(fin)
for fp in audio_file:
    fn = os.path.basename(fp)[:-4]
    text_data.append(ref_dict[fn])

batch_size = 256
audio_embed_list = []
model.eval()
with torch.no_grad():
    for k in tqdm(range(0, len(audio_file), batch_size)):
        audio_embed = model.get_audio_embedding_from_filelist(x = audio_file[k:(k+batch_size)], use_tensor=True)
        audio_embed_list.append(audio_embed.cpu())
        # np.savez('/pscratch/sd/x/xiuliu/vggsound_clap_feat/feat_batch_{}.npz'.format(k), file_names = np.array(audio_file[k:(k+batch_size)]), clap_feat = audio_embed.cpu().numpy())
        # break
# print(audio_embed[:,-20:])
audio_embed = torch.cat(audio_embed_list, dim=0).cuda()
# exit(0)
# Get text embedings from texts, but return torch tensor:

text_embed_list = []
with torch.no_grad():
    for k in tqdm(range(0, len(text_data), batch_size)):
        text_embed = model.get_text_embedding(text_data[k:(k+batch_size)], use_tensor=True)
        text_embed_list.append(text_embed)
text_embed = torch.cat(text_embed_list, dim=0)

# print(text_embed)
print(text_embed.shape)
# np.savez("/pscratch/sd/x/xiuliu/llama_vggsound_gen_audio_caps_clap_feat.npz",  file_names = np.array(audio_file), clap_feat=text_embed.cpu().numpy())

# exit(0)
score = []
for i in range(audio_embed.shape[0]):
    cosine_sim = audio_embed[i:(i + 1)] @ text_embed[i:(i + 1)].t()
    score.append(cosine_sim[0].cpu().item())
print("AVG CLAP score: ", np.mean(np.array(score)))