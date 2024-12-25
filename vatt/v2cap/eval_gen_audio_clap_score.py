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
audio_file_path_1 = "/pscratch/sd/x/xiuliu/vggsound_audio_wav/test/*.wav"
audio_file_1 = glob.glob(audio_file_path_1)


audio_file_path_2 = "/pscratch/sd/x/xiuliu/vt2a_mlm_alibi_mix_large_uni_encodec_vgg_stage_2_wav_no_help/*.wav"


batch_size = 256
audio_1_embed_list = []
model.eval()
with torch.no_grad():
    for k in tqdm(range(0, len(audio_file_1), batch_size)):
        audio_embed_1 = model.get_audio_embedding_from_filelist(x = audio_file_1[k:(k+batch_size)], use_tensor=True)
        audio_1_embed_list.append(audio_embed_1.cpu())
# print(audio_embed[:,-20:])
audio_embed_1 = torch.cat(audio_1_embed_list, dim=0).cuda()
# exit(0)
# Get text embedings from texts, but return torch tensor:

audio_2_embed_list = []
with torch.no_grad():
    for k in tqdm(range(0, len(audio_file_2), batch_size)):
        audio_embed_2 = model.get_audio_embedding_from_filelist(x = audio_file_2[k:(k+batch_size)], use_tensor=True)
        audio_2_embed_list.append(audio_embed_2.cpu())
# print(audio_embed[:,-20:])
audio_embed_2 = torch.cat(audio_2_embed_list, dim=0).cuda()

score = []
for i in range(audio_embed_1.shape[0]):
    cosine_sim = audio_embed_1[i:(i + 1)] @ audio_embed_2[i:(i + 1)].t()
    score.append(cosine_sim[0].cpu().item())
print("AVG CLAP score: ", np.mean(np.array(score)))