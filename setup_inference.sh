# Stage 1: Create the env for VATT in order to generate audio tokens from multi-modal conditions, i.e., video + text (optional)
conda create vatt_env python=3.10
conda activate vatt_env
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/DragonLiu1995/video-to-audio-through-text.git
cd video-to-audio-through-text
pip install --upgrade-strategy only-if-needed -r vatt/requirements.txt
pip install gdown open-clip-torch
pip install -e third_party/hf-dev/transformers-main
pip install -e thrid_party/peft-main


export PYTHONPATH="$PWD:$PWD/third_party:$PWD/vatt:${PYTHONPATH}"

cd vatt
# Download all necessary checkpoint components you might need for either training or inference.
mkdir checkpoints
gdown --id 1rjg-_DKzpxCxX51gwiFD3vYv3iKBPQNM -O vicuna.zip && unzip vicuna.zip -d checkpoints/
gdown --id 1Z-R72RXCapiWUcc35qq8vkzv7U9jrCKu -O checkpoint-117000.zip && unzip checkpoint-117000.zip -d checkpoints/
wget -O vatt_models.zip https://www.dropbox.com/scl/fi/o1663dvgavdtgryltqvmq/vatt_models.zip?rlkey=yzk7o6fb6kpdvs7mnke03t3c0&e=1&st=wr75y7xn&dl=0
unzip vatt_models.zip -d checkpoints/
wget -O audiogen_models.zip https://www.dropbox.com/scl/fi/48nasacz1qikulwv4nj87/audiogen_models.zip?rlkey=3j4beimh2q8h9rw3jmd1mxqsd&e=1&st=lbv24b4c&dl=0
unzip audiogen_models.zip -d checkpoints/
gdown --id 1LdGOtB1s91Lc6Gi45ZD9cSZp6-1qHhLP -O checkpoints/meta_pretrain_vgg_encodec_embed.pt

# Download for necessary data files you might need for training or evaluation purposes.
mkdir dataset
gdown --id 1uo4Hx6tAnqVkU65AfPHGwFAftysTCXxs -O dataset/vggsound_v2a_instruction.json
gdown --id 1Mgb1CWNqL99q4DWh57derAfDdQeOEkBp -O vggsound_clip_5fps.zip && unzip -j vggsound_clip_5fps.zip -d dataset/vggsound_clip_5fps
gdown --id 1_dYe52NcsG0fkvgMa4ixaFQtj1Xwqovv -O meta_pretrain_vgg_encodec_tokens.zip && unzip meta_pretrain_vgg_encodec_tokens.zip -d dataset/

# Replace the pass in scripts, linking to the correct paths of data or checkpoints.
sed -i "s#/path/to/pretrained_mdls/vicuna/#${PWD}/checkpoints/vicuna/#" ./vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml
sed -i "s#/path/to/checkpoint-117000/pytorch_model.bin#${PWD}/checkpoints/checkpoint-117000/pytorch_model.bin#" ./vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml
sed -i "s#/path/to/meta_pretrain_vgg_encodec_embed.pt#${PWD}/checkpoints/meta_pretrain_vgg_encodec_embed.pt#" ./vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml
sed -i "s#/path/to/vggsound_v2a_instruction.json#${PWD}/dataset/vggsound_v2a_instruction.json#" ./vt2a/vt2a_mlm_ff_generation.py
sed -i "s#/path/to/pretrained_mdls/vicuna/#${PWD}/checkpoints/vicuna/#" ./vt2a/vt2a_mlm_ff_generation.py
sed -i "s#/path/to/vggsound_test.txt#${PWD}/vt2a/vggsound_test.txt#"./vt2a/encodec2audio.py
sed -i "s#/path/to/models--facebook--audiogen-medium/snapshots/1277dd7dfd8fa57a205a70acc5de0ee90804502f/#${PWD}/checkpoints/audiogen_model/#" ./vt2a/encodec2audio.py

# Running the script to generate audio tokens using VATT-LLama-T model.
mkdir gen_tokens
cd ..
python ./vatt/vt2a/vt2a_mlm_ff_generation.py -s ./vatt/gen_tokens --img_embs_dir ./vatt/dataset/vggsound_clip_5fps/ --meta_dir ./vatt/vt2a/ \
  --config_file ./vatt/vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml \
  --ckpt_path ./vatt/checkpoints/vatt_models/vatt_llama_T_final.ckpt



# Stage 2: Create the env for audiocraft in order to use Encodec model to decode generated tokens into audio waveforms.
conda create encodec_env python=3.9
conda activate encodec_env
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft  # stable release

# Decode the generated audio tokens back to audio waveforms.
mkdir vatt/gen_wav
python ./vatt/vt2a/encodec2audio.py -s ./vatt/gen_wav --encodec_dir ./vatt/gen_tokens   
