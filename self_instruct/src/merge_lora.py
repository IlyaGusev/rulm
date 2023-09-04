import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


model_name = sys.argv[1]
output_path = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = PeftConfig.from_pretrained(model_name)
base_model_path = config.base_model_name_or_path


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_model = PeftModel.from_pretrained(
    base_model,
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_model = lora_model.merge_and_unload()
lora_model.train(False)

lora_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
