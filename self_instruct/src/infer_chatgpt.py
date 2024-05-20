import json

import fire
from jinja2 import Template
from tqdm import tqdm

from src.util.io import read_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(task=record).strip() + "\n"


def infer_batch(batch, model_name, template_path, output_file):
    prompts = [r["prompt"] for r in batch]
    results = openai_batch_completion(
        batch=prompts,
        model_name=model_name,
        decoding_args=OpenAIDecodingArguments(
            max_tokens=4096
        )
    )
    for r, prompt, result in zip(batch, prompts, results):
        print(prompt)
        print(result.message["content"])
        r["answer"] = result.message["content"]
        output_file.write(json.dumps(r, ensure_ascii=False).strip() + "\n")
    return results


def main(
    input_path,
    output_path,
    template_path,
    model_name,
    request_batch_size=5
):
    records = read_jsonl(input_path)

    batch = []
    with open(output_path, "w") as w:
        for record in tqdm(records):
            batch.append(record)
            if len(batch) != request_batch_size:
                continue
            infer_batch(
                batch=batch,
                model_name=model_name,
                template_path=template_path,
                output_file=w
            )
            batch = []

        if batch:
            infer_batch(
                batch=batch,
                model_name=model_name,
                template_path=template_path,
                output_file=w
            )


if __name__ == "__main__":
    fire.Fire(main)
