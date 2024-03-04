import fire
import os

from llama_lora import Llama_Lora
from falcon_lora import Falcon_Lora
from flan_t5_lora import Flan_t5_Lora
from llama_lora_es import Llama_Lora_es
from umt5_lora import UMT5_Lora

import logging
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
now = datetime.now()


def main(
        base_model_name: str = "clibrain/Llama-2-7b-ft-instruct-es",
        train_data_path: str = "data/legal/files_2023-12-21/val.json",
        ckpt_fie: str = "./ckp_llama2_es_lora_ft",
        batch_size: int = 64,
        micro_batch_size: int = 32,
        num_epochs: int = 3,
        eval_steps: int = 100,
        logging_steps: int = 5,
        warmup_steps: int = 100,
        learning_rate: float = 3e-4,
        val_set_size: int = 8,
        lora_r: int = 256,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        cutoff_length: int = 256,
):
    """
    

    Train a model with LoRA (Low-Rank Adaptation of Large Language Models).

    Parameters
    ----------
    base_model_name : str, optional
        The name of the base model. 
    train_data_path : str, optional
        Path to the training data. 
    ckpt_file : str, optional
        Path to save the checkpoint file.
    batch_size : int, optional
        Batch size for training.
    micro_batch_size : int, optional
        Micro batch size. Default is 32.
    num_epochs : int, optional
        Number of epochs for training.
    eval_steps : int, optional
        Number of evaluation steps.
    logging_steps : int, optional
        Number of logging steps.
    warmup_steps : int, optional
        Number of warmup steps.
    learning_rate : float, optional
        Learning rate.
    val_set_size : int, optional
        Size of the validation set.
    lora_r : int, optional
        LoRA rank parameter.
    lora_alpha : int, optional
        LoRA alpha parameter scaling factor.
    lora_dropout : float, optional
        LoRA dropout rate.
    cutoff_length : int, optional
        Cutoff length for input sequences.

    Raises
    ------
    ValueError
        If the model name is unknown.

    """
    print(f"Fine-tuning {base_model_name} with LoRA ...")
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
    
    m.train(
        data_path=train_data_path, 
        output_dir=ckpt_fie, 
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        val_set_size=val_set_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        wandb_project="uva-nlp-legal",
        wandb_run_name=now.strftime("%Y-%m-%d-%H-%M-%S"),
	)


if __name__ == "__main__":
    fire.Fire(main)
