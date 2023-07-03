import sys

import torch

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


def load_saiga(model_name, use_4bit: bool = False, torch_compile: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        config = PeftConfig.from_pretrained(model_name)
        if use_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            torch_dtype=torch.float16
        )
    elif device == "cpu":
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            device_map={"": device}
        )

    model.eval()
    if torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer, generation_config
