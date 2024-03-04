import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def merge_lora(
    model_name,
    output_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = PeftConfig.from_pretrained(model_name)
    base_model_path = config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    lora_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(merge_lora)
