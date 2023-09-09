import json
import random

import fire
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


JINJA_ENV = Environment(loader=FileSystemLoader("."))


def encode_pair(record, template_path):
    template = JINJA_ENV.get_template(template_path)
    prompt = record["prompt"].strip()
    prompt = prompt.replace("Выход:", "").strip()
    a = record["a"].strip()
    b = record["b"].strip()
    is_reversed = random.random() < 0.5
    if is_reversed:
        a, b = b, a
    record["prediction_is_reversed"] = is_reversed
    return template.render(prompt=prompt, a=a, b=b).strip() + "\n"


def main(
    input_path,
    output_path,
    template_path,
    model_name="gpt-3.5-turbo",
    request_batch_size=5
):
    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    batch = []
    with open(output_path, "w") as w:
        for record in tqdm(records):
            batch.append(record)
            if len(batch) != request_batch_size:
                continue
            prompts = [[{"role": "user", "content": encode_pair(r, template_path)}] for r in batch]
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
                label = result.split("\n")[0].strip().strip('"').lower()
                is_reversed = r.pop("prediction_is_reversed")
                if is_reversed:
                    if "агент 1" in label:
                        label = "агент 2"
                    elif "агент 2" in label:
                        label = "агент 1"
                r["prediction"] = label
                r["explanation"] = "\n".join(result.split("\n")[1:]).strip()
                w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")
            batch = []


if __name__ == "__main__":
    fire.Fire(main)
