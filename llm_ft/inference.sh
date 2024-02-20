#!/bin/bash

data_path=./data/legal/2024-02-04
ckpt_path=./ckp_llama2_es_lora_ft_0204
input_length=128
epoch=3
lora_r=32
lora_alpha=512
data_split=test

# fine-tune
python predict_legal.py \
    --base_model_name clibrain/Llama-2-7b-ft-instruct-es \
    --ckpt_fie ${ckpt_path} \
    --valid_data_path ${data_path}/${data_split}.json \
    --output_file ${data_path}/llama2_es_lora_${data_split}_preds.txt \
    --output_report ${data_path}/report_epoch${epoch}_r${lora_r}_alpha${lora_alpha}_${data_split}.txt \
    --cutoff_length ${input_length} 

# few-shot
python predict_legal.py \
    --base_model_name clibrain/Llama-2-7b-ft-instruct-es \
    --ckpt_fie '' \
    --valid_data_path ${data_path}/${data_split}.json \
    --demo_data_path ${data_path}/shots_for_llama.json \
    --output_file ${data_path}/llama2_es_1shot_${data_split}_preds.txt \
    --output_report ${data_path}/report_1shot_${data_split}.txt \
    --cutoff_length ${input_length} \
    --k_shot 1

# zero-shot
python predict_legal.py \
    --base_model_name google/umt5-xl \
    --ckpt_fie '' \
    --valid_data_path ${data_path}/${data_split}.json \
    --demo_data_path '' \
    --output_file ${data_path}/umt5_zeroshot_${data_split}_preds.txt \
    --output_report ${data_path}/umt5_report_zeroshot_${data_split}.txt \
    --cutoff_length ${input_length} \
    --k_shot 0