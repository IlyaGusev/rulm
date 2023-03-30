import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import peft
from peft import PeftModel, PeftConfig
import torch

model_name = sys.argv[1]
model_type = sys.argv[2]
output_path = sys.argv[3]
model_types = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
}

assert model_type in model_types

config = PeftConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, model_name)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model
model.save_pretrained(output_path)
