"""
flan_t5_lora.py

The class for fine-tuning umt5 with LoRA
"""
import os, sys
import torch, transformers
from typing import List, Union

# The corresponding Transformer module
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# The LLM_Lora base class
from llm_lora import LLM_Lora
from peft import TaskType
from prompter import Prompter

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


class UMT5_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "google/umt5-xxl",
                 prompt_template_name: str = "generic",
                 lora_target_modules: List[str] = ["q","v"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256,
                 task_type: str = TaskType.SEQ_2_SEQ_LM
                 ):
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          prompt_template_name = prompt_template_name,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          task_type = task_type
                        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        
    def load_base_model(self):
        if len(self.base_model) == 0:
            raise ValueError(f"Need to specify a UMT5 pre-trained model -- the current base model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)


    def train(self,
              data_path: str = "",
              output_dir: str = "",
              batch_size: int = 8,
              micro_batch_size: int = 4,
              num_epochs: int = 3,
              eval_steps: int = 5,
              logging_steps: int = 5,
              warmup_steps: int = 0,
              learning_rate: float = 3e-4,
              val_set_size: int = 128,
              lora_r: int = 8,
              lora_alpha: int = 16,
              lora_dropout: float = 0.05,
              add_eos_token: bool = True,
              train_on_inputs: bool = False,
              group_by_length: bool = False,
              wandb_project: str = "",
              wandb_run_name: str = "",
              wandb_watch: str = "",
              wandb_log_model: str = "",
              ):

        print(f"learning rate: {learning_rate}\n")
        # Load the base model
        self.load_base_model()
        # Load the prompter, by default we don't use kshot for traing
        # unless it's MetaICL
        self.prompter = Prompter(kshot=False, cutoff_length=self.cutoff_length)
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token

        # Set up the configuration
        gradient_accumulation_steps = batch_size // micro_batch_size
        
        # =========================================
        # Check if parameter passed or if set within environ
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
        
        # ==========================================
        # Prepare the model for training
        self.model = prepare_model_for_int8_training(self.model)
        self.config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=self.task_type,
        )
        self.model = get_peft_model(self.model, self.config)
        self.model.print_trainable_parameters()
        
        # ==========================================
        # Load data
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = None
        # Ignore the part of resuming from checkpoints
        
        # ===========================================
        # Trainer
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=False,
                logging_steps=logging_steps,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_steps if val_set_size > 0 else None,
                save_steps=eval_steps,
                output_dir=output_dir,
                save_total_limit=3,
                save_safetensors=False, # To avoid the bug in SafeTensors loading module
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
        )
        self.model.config.use_cache = False

        # ===========================================
        # Save model
        # Commented the following block to address the empty file issue
        # old_state_dict = self.model.state_dict
        # self.model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(self.model, type(self.model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        trainer.train()

        self.model.save_pretrained(output_dir)


    def _eval(self,
              instruction: Union[None, str] = None,
              input: Union[None, str] = None,
              demos: Union[None, List[dict]] = None,
              max_new_tokens: int = 128,
              ):
        # Check the values
        if (instruction is None) or (input is None):
            print(f"Instruction: {instruction}\nInput: {input}")
            raise ValueError("Both instruction and input cannot be None")

        # Format the prompt
        prompt = self.prompter.generate_seq2seq_prompt(
            instruction = instruction,
            input = input,
            demos = demos,
            cutoff_length = self.cutoff_length
        )
        
        # Manually cut the input length
        # tokens = prompt.split()
        # if len(tokens) >= self.cutoff_length:
        #     print(f"Prompt is too long, manually truncate it to {self.cutoff_length}")
        #     prompt = " ".join(tokens[:self.cutoff_length])
        # 
        inputs = self.tokenizer(prompt['inputs'], return_tensors="pt", truncation=False)
        if torch.cuda.is_available():
            input_ids = inputs["input_ids"].to("cuda")
        else:
            input_ids = inputs["input_ids"].to("cpu")

        # Inference
        with torch.no_grad():
            gen_output = self.model.generate(
                input_ids = input_ids,
                generation_config = self.gen_config,
                return_dict_in_generate = True, # ?
                output_scores = True,
                max_new_tokens = max_new_tokens,
            )
        s = gen_output.sequences[0] # Get the output token indices
        output = self.tokenizer.decode(s, skip_special_tokens=True) # Map to tokens
        # del prompt, input_ids, gen_output, s
        # torch.cuda.empty_cache()
        return self.prompter.get_response(output) # Only keep the response part


    def tokenize(self, prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt['inputs'],
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        labels = self.tokenizer(
            text_target=prompt['labels'],
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = labels["input_ids"]
        return result


    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_seq2seq_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
            cutoff_length = self.cutoff_length
        )
        # print('full_prompt', full_prompt)
        tokenized_full_prompt = self.tokenize(full_prompt)
        # print('tokenized_full_prompt', tokenized_full_prompt)
        return tokenized_full_prompt