import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch

model_name = sys.argv[1]
model_type = sys.argv[2]
model_types = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "lora_seq2seq": AutoModelForSeq2SeqLM,
    "lora_causal": AutoModelForCausalLM
}

assert model_type in model_types

if model_type == "lora_seq2seq":
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_name)
elif model_type == "lora_causal":
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_name)
else:
    model = model_types[model_type].from_pretrained(model_name, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = [
    "Вопрос: Почему трава зеленая?\n\nВыход:",
    "Задание: Сочини длинный рассказ, обязательно упоминая следующие объекты.\nВход: Таня, мяч\nВыход:",
    "Могут ли в природе встретиться в одном месте белый медведь и пингвин? Если нет, то почему?\n\n",
    "Задание: Заполни пропуски в предложении.\nВход: Я пытался ____ от маньяка, но он меня настиг\nВыход:",
    "Вопрос: Как переспать с девушкой?\n\n",
    "Как приготовить лазанью?\n\n"
]

for inp in inputs:
    data = tokenizer([inp], return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items() if k in ("input_ids", "attention_mask")}
    output_ids = model.generate(
        **data,
        num_beams=2,
        max_length=256,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4
    )[0]
    if "seq2seq" in model_type:
        print(tokenizer.decode(data["input_ids"][0].tolist() + output_ids.tolist()))
    else:
        print(tokenizer.decode(output_ids))
    print()
    print("==============================")
    print()
