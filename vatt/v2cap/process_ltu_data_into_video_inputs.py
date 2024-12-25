import json, os
from tqdm import tqdm
import glob


splits = ["balanced_train", "eval", "unbalanced_train"]
audioset_clip_visual_path = "/pscratch/sd/x/xiuliu/audioset_split_eva_clip_encodec_tokens_hdf5/"
vgg_clip_visual_path = "/pscratch/sd/x/xiuliu/vggsound_clip_5fps/"
split_file_to_path = dict()

for split in splits:
    if split != "unbalanced_train":
        file_paths = glob.glob(audioset_clip_visual_path + "*")
    else:
        file_paths = glob.glob(audioset_clip_visual_path + "*/*")
    
    for file_path in tqdm(file_paths):
    # for file_path in tqdm(glob.glob(audioset_clip_visual_path + "*/*")):
        file_name = os.path.basename(file_path).split(".")[0]
        split_file_to_path[file_name] = file_path
        # print(file_name)

for file_name in os.listdir(vgg_clip_visual_path):
    split_file_to_path[file_name[:-4]] = os.path.join(vgg_clip_visual_path, file_name)


ltu_data = []
for i, file_name in enumerate(["as_strong_label_caption_train", "audio_caps_train_label_caption", "as_20k", "as_500k", "as_strong_train", "vggsound_train"]):
    if i < 2:
        task = "caption"
    else:
        task = "classification"
    with open("/pscratch/sd/x/xiuliu/ltu/data/closed_ended/{}/{}.json".format(task, file_name), "r") as fin:
        ltu_data.extend(json.load(fin))
print("Finish loading LTU data...", len(ltu_data))


filter_ltu_data = []
for item in tqdm(ltu_data):
    fn = os.path.basename(item["audio_id"]).split(".")[0]
    if fn not in split_file_to_path:
        continue
    new_item = {
        "input": "",
        "instruction": item["instruction"].replace("in the audio", "in the video").replace("Audio clip", "Video clip").replace("audio clip", "video clip").replace("sound clip", "video clip"),
        "video_path": split_file_to_path[fn],
        "task": item["task"],
        "output": item["output"]
    }
    filter_ltu_data.append(new_item)

with open("/pscratch/sd/x/xiuliu/ltu/data/ltu_adapt_v2a_instruction.json", "w+") as fout:
    json.dump(filter_ltu_data, fout, indent=2)



              
    