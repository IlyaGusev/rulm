from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = sys.argv[1]


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inp = "Задание: напиши рассказ об объектах.\nВход: Таня, мяч\nОтвет: "
tokens = tokenizer(inp, return_tensors="pt")
print(tokens)
input_ids = tokens["input_ids"][:, :-1]
attention_mask = tokens["attention_mask"][:, :-1]
print(input_ids)
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=128,
    num_beams=1,
    do_sample=True,
    top_p=0.95
)[0]
print(tokenizer.decode(output_ids))
