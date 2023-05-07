import time
import json
import os
import random
import re
import string
import shutil
from functools import partial
from multiprocessing import Pool
from jinja2 import Template

import fire
import numpy as np
import tqdm
from rouge_score import rouge_scorer
import re

from src.util.io import read_jsonl, write_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


NON_ALPHANUM_RE = re.compile(r"[^a-zа-яё0-9]+")

def tokenize(text):
    text = text.lower()
    text = NON_ALPHANUM_RE.sub(" ", text)
    return text.split()


def encode_prompt(example_chars, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    for char in example_chars:
        char.pop("most_similar_chars", None)
        char.pop("avg_similarity_score", None)
    return template.render(
        example_chars=json.dumps(example_chars, ensure_ascii=False)
    ).strip() + "\n"


def post_process(response):
    if not response:
        return []
    if response["finish_reason"] == "length":
        return []
    raw_content = response["message"]["content"]
    try:
        chars = json.loads(raw_content)
        if isinstance(chars, list):
            return chars
        elif isinstance(chars, dict):
            return chars["characters"]
    except Exception:
        return []


def generate_chars(
    output_path,
    seed_chars_path,
    template_path,
    num_chars_to_generate=200,
    model_name="gpt-4",
    request_batch_size=5,
    temperature=1.0,
    top_p=0.95,
    num_cpus=8,
    rouge_cutoff=0.24
):
    random.seed(43)
    seed_chars = [json.loads(l) for l in open(seed_chars_path, "r")]
    print(f"Loaded {len(seed_chars)} character examples")

    machine_chars = []
    if os.path.exists(output_path):
        machine_chars = read_jsonl(output_path)
        print(f"Loaded {len(machine_chars)} machine-generated characters")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_descriptions = [d["context"] for d in seed_chars + machine_chars]
    all_description_tokens = [tokenize(d) for d in all_descriptions]

    request_idx = 0
    progress_bar = tqdm.tqdm(total=num_chars_to_generate)
    if machine_chars:
        progress_bar.update(len(machine_chars))

    is_prompt_printed = False
    is_output_printed = False
    while len(machine_chars) < num_chars_to_generate:
        request_idx += 1

        batch = []
        for _ in range(request_batch_size):
            if machine_chars:
                prompt_chars = random.sample(machine_chars, 1)
                prompt_chars += random.sample(seed_chars, 1)
            else:
                prompt_chars = random.sample(seed_chars, 2)
            random.shuffle(prompt_chars)

            prompt = encode_prompt(prompt_chars, template_path)
            messages = [{"role": "user", "content": prompt}]
            batch.append(messages)

        if not is_prompt_printed:
            is_prompt_printed = True
            print("Prompt example:")
            for message in batch[0]:
                print("Role: {}, content: {}".format(message["role"], message["content"]))

        request_start = time.time()
        results = openai_batch_completion(
            batch=batch,
            model_name=model_name,
            decoding_args=OpenAIDecodingArguments(
                temperature=temperature,
                top_p=top_p
            )
        )
        if not is_output_printed:
            is_output_printed = True
            print("Output example:")
            print(results[0].message["content"])
        request_duration = time.time() - request_start

        process_start = time.time()
        new_chars = []
        for result in results:
            new_chars.extend(post_process(result))

        total = len(new_chars)
        keep = 0
        for new_char in new_chars:
            new_description_tokens = tokenize(new_char["context"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_description_tokens),
                    all_description_tokens,
                )
                rouge_scores = [score.fmeasure for score in rouge_scores]
            if max(rouge_scores) > rouge_cutoff:
                continue

            most_similar_chars = {
                all_descriptions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }

            keep += 1
            new_char["most_similar_chars"] = most_similar_chars
            new_char["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_chars.append(new_char)
            all_descriptions.append(new_char["context"])
            all_description_tokens.append(new_description_tokens)
            progress_bar.update(1)

        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} chars, kept {keep} chars")
        print("===================================")

        write_jsonl(machine_chars, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(generate_chars)
