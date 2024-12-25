from transformers import VideoEncoder, GenerationConfig
import torch
from datasets import load_dataset



from transformers import GemmaTokenizer, GemmaForCausalLM, GemmaModel, GemmaConfig
tokenizer = GemmaTokenizer.from_pretrained("../../pretrained_mdls/gemma_2b/")
# gemma_2b_config = GemmaConfig(
#     num_hidden_layers=18,
#     num_attention_heads=8,
#     num_key_value_heads=1,
#     hidden_size=2048,
#     intermediate_size=16384,
# )
model = GemmaForCausalLM.from_pretrained("../../pretrained_mdls/gemma_2b/")
model.cuda()
prompt = "Compute 1 + 3 = ?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
exit(0)

if __name__ == "__main__":
    # net = VideoEncoder()
    # net.cuda()
    # x = torch.rand(64, 50, 768).cuda()
    # out = net(x)
    # print(out.shape)
    import json
    with open("../../data/vggsound_v2a_instruction.json", "r") as fin:
        data = json.load(fin)
    count = 0
    for item in data:
        if "video_path" not in item:
            print(item)
            count += 1
        if item["split"] not in ["train", "test"]:
            print(item)
            count += 1
    print(count)
    exit(0)

    data = load_dataset("json", data_files="../../data/vggsound_v2a_instruction.json")
    print(len(data["train"]))#, len(data["test"]))
    print(data["train"][0])
    exit(0)


    from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
    tokenizer = LlamaTokenizer.from_pretrained("../../pretrained_mdls/vicuna/")
    config = LlamaConfig()
    model = LlamaForCausalLM(config)
    model.cuda()


   

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    tokenizer.padding_side = "left"  # Allow batched inference

    prompt = "Describe the audio associated with the video:"
    inputs = tokenizer(prompt, return_tensors="pt")
    video_input = torch.rand(2, 50, 768).cuda()

    temp, top_p, top_k = 0.1, 0.95, 500


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
        max_new_tokens=30,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.cuda(), video_input, generation_config=generation_config, max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
