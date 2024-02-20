import fire

from llama_lora import Llama_Lora
from falcon_lora import Falcon_Lora
from flan_t5_lora import Flan_t5_Lora

import logging

logging.basicConfig(level=logging.ERROR)


def main(
        base_model_name: str = "meta-llama/Llama-2-7b-hf", # # "google/flan-t5-base", # "tiiuae/falcon-7b",
        output_dir: str = "",
):
    print(f"Fine-tuning {base_model_name} with LoRA ...")
    if 'Llama' in base_model_name:
        m = Llama_Lora(
            base_model=base_model_name,
        )
        # m.train(
        #     data_path="data/legal/legal-small.json",
        #     output_dir="./ckp_llama2_lora_legal" if len(output_dir) == 0 else output_dir,
        #     num_epochs=3,
        #     group_by_length=False,
        # )
        m.predict(
            input_file = "data/legal/legal-small.json",
            lora_adapter = "./ckp_llama2_lora_legal",
	        label_set = ["Factual Background", "Prior Rulings", "Court's Analysis","Challenged Statute(s)",
              "Plaintiff's Allegations", "Decision", "Third-Party Intervention", "Summary of Findings", "Exhibits",
              "Third-Party Intervention: Procurador","Jurisdiction", "Hearing"],
            # input_file = "data/sst2_json/sst2_dev.json",
            # label_set = ["positive", "negative"],
            kshot = 2,
            demo_file = "data/sst2_json/sst2_dev.json",
	    max_new_tokens = 6,
            verbose = True,
        )
    elif 'falcon' in base_model_name:
        m = Falcon_Lora(
            base_model=base_model_name,
        )
        m.train(
            data_path="data/sst2_json/sst2_dev.json",
            output_dir="./ckp_falcon_lora" if len(output_dir) == 0 else output_dir,
            num_epochs=10,
            group_by_length=False,
        )
    elif 'flan' in base_model_name:
        m = Flan_t5_Lora(
            base_model=base_model_name,
        )
        m.train(
            data_path="data/sst2_json/sst2_dev.json",
            output_dir="./ckp_flan_t5_lora" if len(output_dir) == 0 else output_dir,
            num_epochs=10,
            group_by_length=False,
        )
    else:
        raise ValueError(f"Unknown model name: {base_model_name}")


if __name__ == "__main__":
    fire.Fire(main)
