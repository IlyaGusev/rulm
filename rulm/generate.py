from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = sys.argv[1]


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'

inp = ""
tokens = tokenizer(inp, return_tensors="pt")
input_ids = tokens["input_ids"][:, :-1]
attention_mask = tokens["attention_mask"][:, :-1]
position_ids = attention_mask.cumsum(-1)
token_type_ids = tokens["token_type_ids"][:, :-1]
print(tokens)
output_ids = model.generate(
    input_ids=input_ids,
    #attention_mask=attention_mask, token_type_ids=token_type_ids,
    max_length=64, num_beams=1, do_sample=True, top_p=0.95
)[0]
print(tokenizer.decode(output_ids))
