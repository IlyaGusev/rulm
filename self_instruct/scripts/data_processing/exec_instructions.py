import json
import os

import fire
from jinja2 import Template
from tqdm import tqdm

import utils


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(task=record).strip() + "\n"


def main(
    input_path,
    output_path,
    template_path,
    existing_path=None,
    model_name="gpt-3.5-turbo",
    request_batch_size=5
):
    existing_keys = set()
    if existing_path and os.path.exists(existing_path):
        with open(existing_path) as f:
            existing_records = [json.loads(line) for line in f]
            existing_records = [r for r in existing_records if "label" in r and r["label"]]
            existing_keys = {tuple((r["instruction"], r["input"])) for r in existing_records}

    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    batch = []
    with open(output_path, "w") as w:
        for record in tqdm(records):
            key = tuple((record["instruction"], record["input"]))
            if key in existing_keys:
                continue
            if "noinput" in record["input"]:
                record["input"] = ""
            batch.append(record)
            if len(batch) != request_batch_size:
                continue
            prompts = [[{"role": "user", "content": encode_prompt(r, template_path)}] for r in batch]
            results = utils.openai_batch_completion(
                batch=prompts,
                model_name=model_name,
                decoding_args=utils.OpenAIDecodingArguments(
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
                r["new_output"] = result
                w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")
            batch = []


if __name__ == "__main__":
    fire.Fire(main)
