from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = sys.argv[1]


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'

inp = "И грянул"
tokens = tokenizer(inp, return_tensors="pt")["input_ids"][:, :-1]
print(tokens)
output_ids = model.generate(
    input_ids=tokens, max_length=64, num_beams=1, do_sample=True, top_p=0.95
)[0]
print(tokenizer.decode(output_ids))
