import json
import os
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from src.util.io import read_jsonl, write_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(char, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(char_context=char["context"]).strip() + "\n"


def get_char_key(char):
    return (char["name"].strip(), char["context"].strip())


def process_batch(batch, model_name, template_path):
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
    final_prompts = dict()
    for char, prompt, result in zip(batch, prompts, results):
        result = result.message["content"]
        final_prompts[get_char_key(char)] = result
        print(prompt[-1]["content"])
        print(result)
        print("=============")
        print()
    return final_prompts


def main(
    chars_path,
    output_path,
    template_path,
    model_name="gpt-4",
    request_batch_size=5
):
    existing_keys = set()
    output_records = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            output_records = [json.loads(line) for line in f]
            existing_keys = {get_char_key(r) for r in output_records}
    print(f"Existing keys: {len(existing_keys)}")

    chars = read_jsonl(chars_path)
    key2idx = {get_char_key(char): idx for idx, char in enumerate(chars)}
    batch = []

    output_chars = []
    for char in tqdm(chars):
        key = get_char_key(char)
        if key in existing_keys:
            print(f"Skipping {key}")
            output_chars.append(char)
            continue
        batch.append(char)
        if len(batch) != request_batch_size:
            continue

        prompts = process_batch(batch, model_name, template_path)
        for key, prompt in prompts.items():
            chars[key2idx[key]]["image_prompt"] = prompt
            output_chars.append(chars[key2idx[key]])

        batch = []
        write_jsonl(output_chars, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)

    if batch:
        prompts = process_batch(batch, model_name, template_path)
        for key, prompt in prompts.items():
            chars[key2idx[key]]["image_prompt"] = prompt
            output_chars.append(chars[key2idx[key]])

        write_jsonl(output_chars, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
