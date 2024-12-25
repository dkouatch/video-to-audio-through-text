# this code is modified from lora_alpaca https://github.com/tloen/alpaca-lora under Apache-2.0 license
import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, Dataset
import numpy as np
print("Start script")

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

vgg_clip_visual_path = "/pscratch/sd/x/xiuliu/vggsound_clip_5fps/"


v2a_instruction_templates = [
    "Imagine possible sounds for this video.",
    "Describe possible sounds that could match the video.",
    "What audio could be inferred from this video?",
    "What sounds could match the video?",
    "Infer the sounds that match the video.",
    "What audio could best match this video?",
    "What sound events could the video yields?",
    "Caption possible sound events that describe the video.",
    "What sound events make sense to this video?",
    "Imagine audio events that match this video.",
]


visual_instruction_templates = [
    "Describe the visual scene.",
    "Caption the visual scene.",
    "What do you see from the video?",
    "What visual content is shown in the video?",
    "Describe the video content in a sentence.",
    "What do you see from the visual frames?",
    "Provide visual description for the scene.",
    "Describe the visual content in details.",
    "What visual elements do you see in the video?",
    "Describe what happens in the video.",
]

def split_train_val(data):
    train_set = set()
    test_set = set()
    train_data = []
    test_data = []
    for item in data["train"]:
        new_item = {
            "video_path": item["video_path"],
            "output": item["output"],
        }
        if item["dataset"] == "vggsound":
            new_item =  {
                        "video_path": os.path.join(vgg_clip_visual_path, item["video_path"]) + ".npy",
                        "output": item["output"],
            }
        if "instruction" in item:
            new_item["instruction"] = item["instruction"]
        if "task" not in item:
            new_item["task"] = "audio_caption" if "caption" not in item else "visual_caption"
        
        if item["split"] == "test":
            test_set.add(item["video_path"])
            test_data.append(new_item)
        else:
            train_set.add(item["video_path"])
            train_data.append(new_item)
    return train_data, test_data, train_set, test_set

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "v2a_llm",
    wandb_run_name: str = "",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    save_steps: int = 100,
    trainable_params = 'all'
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # trick to load checkpoints correctly from HF
    if '../../../pretrained_mdls/vicuna_ltu/' not in base_model:
        # start from a different model with original vicuna
        # temporally first load the original vicuna, then load the actual checkpoint
        start_model = base_model # need to point to a specific bin file that contains state dict.
        # TODO: change to your vicuna_tltr path
        # base_model = '../../../pretrained_mdls/vicuna_ltu/'
        # print('Will load from {:s} later, for implementation purpose, first load from {:s}'.format(start_model, base_model))
    else:
        start_model = None

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    # world_size = int(torch.cuda.device_count())
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/",
        load_in_8bit=False,
        #torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained("/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        if "instruction" in data_point:
            instruction = data_point["instruction"]
        elif data_point["task"] == "audio_caption":
            instruction = np.random.choice(v2a_instruction_templates)
        else:
            instruction = np.random.choice(visual_instruction_templates)
        full_prompt = prompter.generate_prompt(
            instruction,
            "",
            data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction, ""
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # for video params, lora always trainable, llama always frozen
    for name, param in model.named_parameters():
        if trainable_params == 'all':
            if "video" in name:
                param.requires_grad = True
                #print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            if "bert" in name:
                param.requires_grad = True
            if "query_tokens" in name:
                param.requires_grad = True
        if trainable_params == 'proj':
            if "video_proj" in name:
                param.requires_grad = True
                #print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # now load from real checkpoint
    if start_model != None and os.path.isfile(start_model) and (resume_from_checkpoint == None or resume_from_checkpoint == False):
        state_dict = torch.load(start_model, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint', msg)

    # for key, val in model.state_dict().items():
    #     # val.requires_grad = True
    #     print(key)
    #     print(val)
    #     print(val.requires_grad)
    #     print("x"*10)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    # exit(0)
    # if val_set_size > 0:
        # train_val = data["train"].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
    train_data, test_data, train_set, test_set = split_train_val(data)
    train_data = Dataset.from_list(train_data)
    test_data = Dataset.from_list(test_data)
    train_data = (
        train_data.shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        test_data.shuffle().map(generate_and_tokenize_prompt)
    )

    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    ddp = False
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            dataloader_num_workers=2,
            output_dir=output_dir,
            save_total_limit=50,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq2(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)