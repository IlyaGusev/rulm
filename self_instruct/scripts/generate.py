from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import sys

model_name = sys.argv[1]
model_type = sys.argv[2]
model_types = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM
}

assert model_type in model_types

model = model_types[model_type].from_pretrained(model_name).to("cuda")
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
    print(data)
    data = {k: v.to("cuda") for k, v in data.items() if k in ("input_ids", "attention_mask")}
    output_ids = model.generate(
        **data,
        num_beams=1,
        max_length=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4
    )[0]
    print(tokenizer.decode(output_ids))
    print()
    print("==============================")
    print()
