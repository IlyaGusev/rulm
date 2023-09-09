import os
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from src.util.io import read_jsonl, write_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(task=record).strip() + "\n"


def process_batch(batch, model_name, template_path):
    prompts = [[{"role": "user", "content": encode_prompt(r, template_path)}] for r in batch]
    results = openai_batch_completion(
        batch=prompts,
        model_name=model_name,
        decoding_args=OpenAIDecodingArguments(
            max_tokens=3076
        )
    )
    output_records = []
    for r, prompt, result in zip(batch, prompts, results):
        result = result.message["content"]
        print(prompt[-1]["content"])
        print(result)
        print()
        print("=============")
        print()
        if "NO" in result:
            continue
        r["output"] = result
        output_records.append(r)
    return output_records


def main(
    input_path,
    output_path,
    template_path,
    model_name="gpt-3.5-turbo",
    request_batch_size=5
):
    existing_keys = set()
    output_records = list()
    if output_path and os.path.exists(output_path):
        output_records = read_jsonl(output_path)
        existing_keys = {tuple((r["instruction"].strip(), r["input"].strip())) for r in output_records}
    print(f"Existing keys: {len(existing_keys)}")

    batch = []
    records = read_jsonl(input_path)
    for record in tqdm(records):
        if "noinput" in record["input"]:
            record["input"] = ""
        key = tuple((record["instruction"].strip(), record["input"].strip()))
        if key in existing_keys:
            continue
        batch.append(record)
        if len(batch) != request_batch_size:
            continue

        output_records += process_batch(batch, model_name, template_path)
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)
        batch = []

    if batch:
        output_records += process_batch(batch, model_name, template_path)
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
