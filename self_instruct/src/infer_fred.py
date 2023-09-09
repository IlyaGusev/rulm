import sys
import json
from tqdm import tqdm

import fire
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM

from src.util.io import read_jsonl
from src.util.dl import gen_batch


def generate(model, tokenizer, prompts, generation_config):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("<extra_id_0>", "").strip()
        outputs.append(sample_output)
    return outputs


def generate_answers(
    model_name: str,
    input_path: str,
    output_path: str,
    batch_size: int = 1
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    records = read_jsonl(input_path)
    prompts = []
    template = "<SC6>Человек: {prompt}\nБот: <extra_id_0>"
    for record in records:
        user_message = record["instruction"]
        if "input" in record and record["input"]:
            user_message += "\nДано: " + record["input"]
        prompt = template.format(prompt=user_message)
        prompts.append(prompt)

    all_outputs = []
    for batch in tqdm(gen_batch(prompts, batch_size)):
        outputs = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=batch,
            generation_config=generation_config
        )
        for prompt, output in zip(batch, outputs):
            print(prompt)
            print(output)
            all_outputs.append(output)

    with open(output_path, "w") as w:
        for record, output in zip(records, all_outputs):
            try:
                record["answer"] = output
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
            except Exception:
                record["answer"] = ""
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_answers)
