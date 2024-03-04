"""
falcon_lora.py

The class for fine-tuning Falcon with LoRA
"""
import os
import torch, transformers
from typing import List

# The corresponding Transformer module
from transformers import AutoModelForCausalLM, AutoTokenizer

# The LLM_Lora base class
from llm_lora import LLM_Lora


class Falcon_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 prompt_template_name: str = "generic",
                 lora_target_modules: List[str] = ["query_key_value","dense"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256
                 ):
        """
        Initializes the Falcon_Lora model.

        Parameters
        ----------
        base_model : str, optional
            The name of the base model to be used.
        prompt_template_name : str, optional
            The name of the prompt template to be used.
        lora_target_modules : List[str], optional
            A list of target modules: "query_key_value" and "dense".
        load_in_8bit : bool, optional
            Whether to load the model in 8-bit precision.
        cutoff_length : int, optional
            The maximum length at which inputs are truncated.


        """
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          prompt_template_name = prompt_template_name,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def load_base_model(self):
        """
        Loads the Falcon pre-trained model.

        Raises
        ------
        ValueError
            If no base model is specified.

        """
        if len(self.base_model) == 0:
            raise ValueError(f"Need to specify a Falcon pre-trained model -- the current base model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.padding_side = "left"
        # print(self.model.parameters)
        # sys.exit()

        
