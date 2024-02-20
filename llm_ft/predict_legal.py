import fire
import os
import json
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from llama_lora import Llama_Lora
from falcon_lora import Falcon_Lora
from flan_t5_lora import Flan_t5_Lora
from llama_lora_es import Llama_Lora_es
from umt5_lora import UMT5_Lora

import logging

logging.basicConfig(level=logging.ERROR)


def main(
        base_model_name: str = "clibrain/Llama-2-7b-ft-instruct-es",
        ckpt_fie: str = "",
        valid_data_path: str = "data/legal/files_2023-12-21/val.json",
        demo_data_path: str = "data/legal/files_2023-12-21/shots_for_llama.json",
        output_file: str = "data/legal/llama2_es_lora_ft_val_preds.txt",
        output_report: str = "data/legal/accuracy_epoch3_r256_alpha256.txt",
        cutoff_length: int = 256,
        k_shot: int = 0,
):
    # print(f"Fine-tuning {base_model_name} with LoRA ...")
    if 'llama' in base_model_name.lower():
        m = Llama_Lora_es(
            base_model=base_model_name,
            cutoff_length=cutoff_length,
        )
    elif 't5' in base_model_name.lower():
        m = UMT5_Lora(
            base_model=base_model_name,
            cutoff_length=cutoff_length,
        )
    else:
        raise ValueError(f"Unknown model name: {base_model_name}")
        
    pred_labels = m.predict(
        input_file = valid_data_path,
        lora_adapter = ckpt_fie,
        label_set = ["encabezado", "antecedentes", "pretensiones", "intervenciones", "intervención del procurador", "norma(s) demandada(s)", "actuaciones en sede revisión","pruebas", "audiencia(s) pública(s)", "competencia", "consideraciones de la corte", "síntesis de la decisión", "decisión", "firmas", "salvamento de voto", "sin sección"],
        max_new_tokens = 10,
        kshot = k_shot,
        demo_file = demo_data_path,
        verbose = True,
    )
    with open(output_file, 'w') as fout:
        for text in pred_labels:
            fout.write(text.lower().strip()+'\n')
    # with open(output_file) as f:
    #     preds = f.readlines()
    # pred_labels = [pred.lower().strip() for pred in preds]

    with open(valid_data_path) as f:
        labels = json.load(f)
    gt_labels = [label['output'].lower().strip() for label in labels]
    resport = classification_report(gt_labels, pred_labels, labels=["encabezado", "antecedentes", "pretensiones", "intervenciones", "intervención del procurador", "norma(s) demandada(s)", "actuaciones en sede revisión","pruebas", "audiencia(s) pública(s)", "competencia", "consideraciones de la corte", "síntesis de la decisión", "decisión", "firmas", "salvamento de voto", "sin sección", "none"])
    print(resport)
    accuracy = accuracy_score(gt_labels, pred_labels)
    print(accuracy)
    with open(output_report, 'w') as f:
        f.write(resport)
        f.write(f'\naccuracy: {accuracy}')


if __name__ == "__main__":
    fire.Fire(main)
