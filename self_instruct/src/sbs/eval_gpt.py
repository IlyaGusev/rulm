import json
import random
from collections import defaultdict, Counter

import fire
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from src.util.dl import gen_batch
from src.util.io import read_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


JINJA_ENV = Environment(loader=FileSystemLoader("."))


def encode_pair(record, template_path):
    template = JINJA_ENV.get_template(template_path)
    instruction = record["instruction"].strip()
    left_answer = record["left_answer"].strip()
    right_answer = record["right_answer"].strip()
    return template.render(
        instruction=instruction,
        left_answer=left_answer,
        right_answer=right_answer
    ).strip() + "\n"


def parse_result(result):
    start_index = result.find("{")
    end_index =  result.find("}")
    grades = result[start_index:end_index+1].strip()
    grades = json.loads(grades)
    left_grade, right_grade = grades["a_score"], grades["b_score"]
    left_grade = float(left_grade)
    right_grade = float(right_grade)
    explanation = result[:start_index]
    return left_grade, right_grade, explanation


def main(
    input_path,
    output_path,
    template_path,
    model_name="gpt-3.5-turbo",
    request_batch_size=5
):
    records = read_jsonl(input_path)

    agg_scores = defaultdict(lambda: Counter())
    with open(output_path, "w") as w:
        for batch in gen_batch(records, batch_size=request_batch_size):
            prompts = [[{"role": "user", "content": encode_pair(r, template_path)}] for r in batch]
            results = openai_batch_completion(
                batch=prompts,
                model_name=model_name,
                decoding_args=OpenAIDecodingArguments(
                    max_tokens=2048
                )
            )
            for r, prompt, result in zip(batch, prompts, results):
                result = result.message["content"]
                print(prompt[-1]["content"])
                print(result)
                print()
                print("=============")
                print()
                left_grade, right_grade, explanation = parse_result(result)
                print(left_grade, right_grade, explanation)
                r["left_grade"] = left_grade
                r["right_grade"] = right_grade
                r["explanation"] = explanation
                left_model = r["left_model"]
                right_model = r["right_model"]
                pair = tuple(sorted([left_model, right_model]))
                if left_grade == right_grade:
                    agg_scores[pair]["equal"] += 1
                elif left_grade > right_grade:
                    agg_scores[pair][left_model] += 1
                else:
                    agg_scores[pair][right_model] += 1

                print(agg_scores)
                w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
