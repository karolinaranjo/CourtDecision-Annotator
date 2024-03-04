"""
llama_lora.py

The class for fine-tuning Llama with LoRA
"""

import torch, transformers
from typing import List

# For Llama
from transformers import LlamaForCausalLM, LlamaTokenizer

# The LLM_Lora base class
from llm_lora import LLM_Lora

class Llama_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 prompt_template_name: str = "generic",
                 lora_target_modules: List[str] = ["q_proj", "v_proj"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256,
                 ):
        """
        Initialize the Llama Lora models.

        Parameters
        ----------
        base_model : str, optional
            The name or path of the base language model.
        prompt_template_name : str, optional
            The name of the prompt template.
        lora_target_modules : List[str], optional
            The target modules for LoRA. The default is ["q_proj", "v_proj"].
        load_in_8bit : bool, optional
            Whether to load the model in 8-bit mode.
        cutoff_length : int, optional
            The cutoff length for tokenization. The default is 256.

        """
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          prompt_template_name = prompt_template_name,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          )



    def load_base_model(self):
        """
        Loads the base language model for preparation.
    
        Raises
        ------
        ValueError
            If the base_model attribute is empty
        """
        # Load the model for preparation
        if len(self.base_model) == 0:
            raise ValueError(f"The base_model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"  # Allow batched inference
        # print(self.model.parameters)
        # sys.exit()

        # For the generation model
        # Make it model specific
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
