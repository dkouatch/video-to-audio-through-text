import torch
import numpy as np
import argparse, os
from os.path import join as pjoin
from tqdm import tqdm
import soundfile as sf
import codecs as cs
from audiocraft.models import AudioGen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save waveform output from codec",
        default="/path/to/save_wav_files/"
    )
    parser.add_argument(
        "--encodec_dir",
        type=str,
        required=False,
        help="The path to generated codec data dir",
        default='/path/to/save_gen_codec/'
    )
    args = parser.parse_args()
    save_path = args.save_path
    encodec_dir = args.encodec_dir
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    audiogen = AudioGen.get_pretrained('/path/to/models--facebook--audiogen-medium/snapshots/1277dd7dfd8fa57a205a70acc5de0ee90804502f/') #'facebook/audiogen-medium')
    encodec = audiogen.compression_model
    encodec = encodec.to(device)
    data = []
    with cs.open('/path/to/vggsound_test.txt', "r") as f:
        for line in f.readlines():
            data.append(line.strip())
    print('number of audio files: ', len(data))
    all_time = []
    for fn in tqdm(data):
        save_audio_fn = "{}.wav".format(fn)
        save_audio_path = os.path.join(save_path, save_audio_fn)
        if not os.path.exists(pjoin(encodec_dir, fn + ".npy")): continue
        encodec_tokens = np.load(pjoin(encodec_dir, fn + ".npy"))
        with torch.no_grad():
            audio = encodec.decode(torch.from_numpy(encodec_tokens).unsqueeze(0).to(device))
        sf.write(save_audio_path, audio.cpu().numpy().reshape(-1, 1), 16000)

