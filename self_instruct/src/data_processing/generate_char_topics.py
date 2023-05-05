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
    char.pop("most_similar_chars", None)
    char.pop("avg_similarity_score", None)
    return template.render(char_json=json.dumps(char, ensure_ascii=False)).strip() + "\n"


def get_char_key(char):
    return (char["name"].strip(), char["context"].strip())


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
    batch = []
    for char in tqdm(chars):
        key = get_char_key(char)
        if key in existing_keys:
            print(f"Skipping {key}")
            continue
        batch.append(char)
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
        for char, prompt, result in zip(batch, prompts, results):
            result = result.message["content"]
            topics = result.split("\n")
            cleaned_topics = []
            for topic in topics:
                topic = topic.strip()
                if not topic:
                    continue
                if not topic[0].isnumeric():
                    continue
                topic = " ".join(topic.strip().split(" ")[1:])
                cleaned_topics.append(topic)
            print(prompt[-1]["content"])
            print(cleaned_topics)
            print()
            print("=============")
            print()
            char["topics"] = cleaned_topics
            output_records.append(char)

        batch = []

        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
