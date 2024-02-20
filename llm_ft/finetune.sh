#!/bin/bash

lora_r_list=(
32
# 128
# 256
# 512
)

lora_alpha_list=(
# 64
# 128
# 256
512
)

data_path=./data/legal/2024-02-12
ckpt_path=./ckp_umt5_lora_ft_0212
input_length=128
epoch=3


for lora_r in "${lora_r_list[@]}"; do
    for lora_alpha in "${lora_alpha_list[@]}"; do
        python finetune_legal.py \
            --base_model_name clibrain/Llama-2-7b-ft-instruct-es \
            --train_data_path ${data_path}/train.json \
            --ckpt_fie ${ckpt_path} \
            --batch_size 4 \
            --micro_batch_size 2 \
            --num_epochs ${epoch} \
            --lora_r ${lora_r} \
            --lora_alpha ${lora_alpha} \
            --cutoff_length ${input_length}
        python predict_legal.py \
            --base_model_name clibrain/Llama-2-7b-ft-instruct-es \
            --ckpt_fie ${ckpt_path} \
            --valid_data_path ${data_path}/val.json \
            --output_file ${data_path}/umt5_lora_val_preds.txt \
            --output_report ${data_path}/umt5_report_epoch${epoch}_r${lora_r}_alpha${lora_alpha}_val.txt \
            --cutoff_length ${input_length}
    done
done