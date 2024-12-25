#!/bin/bash

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

output_dir='../exp/visual_audio_finetune_vgg'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=1 --master_port=1235 ../pretrain_visual_audio.py \
    --base_model '/pscratch/sd/x/xiuliu/ltu/src/ltu/exp/visual_finetune_vgg/checkpoint-10000/pytorch_model.bin' \
    --data_path '../../../data/vggsound_all_instruction.json' \
    --output_dir $output_dir \
    --batch_size 32 \
    --micro_batch_size 32 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --wandb_run_name ${output_dir} \
    --group_by_length \
    --save_steps 500 \
    --trainable_params all