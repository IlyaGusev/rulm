import json
import os
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(seed=record["seed"].strip()).strip() + "\n"


def main(
    seeds_path,
    output_path,
    template_path,
    model_name="gpt-3.5-turbo",
    request_batch_size=5
):
    existing_keys = set()
    output_records = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            output_records = [json.loads(line) for line in f]
            existing_keys = {r["seed"].strip() for r in output_records}
    print(f"Existing keys: {len(existing_keys)}")

    with open(seeds_path) as f:
        seeds = [json.loads(line) for line in f]

    batch = []
    for record in tqdm(seeds):
        key = record["seed"].strip()
        if key in existing_keys:
            print(f"Skipping {key}")
            continue
        batch.append(record)
        if len(batch) != request_batch_size:
            continue
        prompts = [[
            {"role": "user", "content": encode_prompt(r, template_path)}
        ] for r in batch]
        results = openai_batch_completion(
            batch=prompts,
            model_name=model_name,
            decoding_args=OpenAIDecodingArguments(
                max_tokens=3076
            )
        )
        for r, prompt, result in zip(batch, prompts, results):
            result = result.message["content"]
            print(prompt[-1]["content"])
            print(result)
            print()
            print("=============")
            print()
            r["output"] = result
            output_records.append(r)

        batch = []

        with open(output_path + "_tmp", "w") as w:
            for record in output_records:
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
