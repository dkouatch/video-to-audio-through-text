#!/bin/bash

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

output_dir='../exp/v2a_caption_gemma'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=1 --master_port=1234 ../finetune_v2a_gemma.py \
    --base_model '../../../pretrained_mdls/gemma_2b/' \
    --data_path '../../../data/vggsound_v2a_instruction.json' \
    --output_dir $output_dir \
    --batch_size 12 \
    --micro_batch_size 12 \
    --num_epochs 4 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ['dummy'] \
    --train_on_inputs \
    --wandb_run_name ${output_dir} \
    --group_by_length \
    --save_steps 500 \
    --trainable_params all