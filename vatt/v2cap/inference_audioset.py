import sys
import os, argparse
from tqdm import tqdm
from glob import glob
import torchaudio
import json
import torch
import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank

def parse_arguments():
    parser = argparse.ArgumentParser(description='sample')
    parser.add_argument("--train",  type=str, default="unbalanced")
    parser.add_argument("--slice", type=int)
    v = vars(parser.parse_args())
    return v

if __name__ == "__main__":
    with open("/global/homes/x/xiuliu/audioset_unbalanced_train_audio_caption_part_first_half.json", "r") as fin:
       exist_caption = json.load(fin)
    exist_set = set(list(exist_caption.keys()))
    hps = parse_arguments()
    is_train = hps["train"]
    print(is_train)
    slice = hps["slice"]


    prompt_template = "alpaca_short" # The prompt template to use, will default to alpaca.

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained('../../pretrained_mdls/vicuna_ltu/')

    model = LlamaForCausalLM.from_pretrained('../../pretrained_mdls/vicuna_ltu/', device_map="auto", torch_dtype=torch.float16)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    exit(0)
    temp, top_p, top_k = 0.1, 0.95, 500
    # change it to your model path
    eval_mdl_path = '../../pretrained_mdls/ltu_ori_paper.bin'
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    instruction = 'Close-ended question: Write an audio caption describing the sound.'
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        max_new_tokens=400,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )


    begin_time = time.time()
    audio_path_base = None
    if is_train == "unbalanced":
        audio_path_base = "/pscratch/sd/x/xiuliu/audioset_audio_data/audios/unbalanced_train_segments/"
    elif is_train == "balanced":
        audio_path_base = "/pscratch/sd/x/xiuliu/audioset_audio_data/audios/balanced_train_segments/"
    else: 
        audio_path_base = "/pscratch/sd/x/xiuliu/audioset_audio_data/audios/eval_segments/"
    all_files = glob(audio_path_base + "*/*.wav")
    if is_train != "unbalanced":
        all_files = glob(audio_path_base + "/*.wav") 
    files_load = all_files
    #if is_train == "unbalanced" and slice < 7:
    #    files_load = all_files[int(slice * len(all_files) // 8): int((slice + 1) * len(all_files) // 8)]
    #elif is_train == "unbalanced" and slice == 7:
    #    files_load = all_files[int(slice * len(all_files) // 8):]

    os.makedirs("/pscratch/sd/x/xiuliu/audioset_audio_caption/", exist_ok=True)
    os.makedirs("/pscratch/sd/x/xiuliu/audioset_audio_caption/unbalanced_train/", exist_ok=True)
    os.makedirs("/pscratch/sd/x/xiuliu/audioset_audio_caption/balanced_train/", exist_ok=True)
    os.makedirs("/pscratch/sd/x/xiuliu/audioset_audio_caption/eval/", exist_ok=True)

    save_folder = "eval"
    if is_train == "unbalanced":
        save_folder = "unbalanced_train"
    elif is_train == "balanced":
        save_folder = "balanced_train"

    prompt = prompter.generate_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    for audio_path in tqdm(files_load):
        try:
            audio_name = os.path.basename(audio_path)

            full_save_path = "/pscratch/sd/x/xiuliu/audioset_audio_caption/{}/{}.txt".format(save_folder, audio_name[:-4])
            if is_train == "unbalanced":
                part_name = audio_path.split("/")[-2]
                full_save_dir = "/pscratch/sd/x/xiuliu/audioset_audio_caption/{}/{}/".format(save_folder, part_name)
                os.makedirs(full_save_dir, exist_ok=True)
                full_save_path = full_save_dir + "{}.txt".format(audio_name[:-4])
            if os.path.exists(full_save_path): continue
            if audio_name[:-4] in exist_set: continue
            
            cur_audio_input = load_audio(audio_path).unsqueeze(0)
            if torch.cuda.is_available() == False:
                pass
            else:
                cur_audio_input = cur_audio_input.half().to(device)

            # Without streaming
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids.to(device),
                    audio_input=cur_audio_input,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=400,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)[6:-4]
            end_time = time.time()
            # print(output)
            # print('eclipse time: ', end_time-begin_time, ' seconds.')

            pred = output[len(prompt):]

            with open(full_save_path, "w+") as fout:
                fout.write(pred)
        except Exception as e:
            print(e)
            print("Problem with file: {}".format(audio_name))
ls 