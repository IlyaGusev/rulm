import time
import json
import os
import random
import re
import string
import shutil
from functools import partial
from multiprocessing import Pool
from jinja2 import Environment, FileSystemLoader

import fire
import numpy as np
import tqdm
from rouge_score import rouge_scorer
import re

import utils


JINJA_ENV = Environment(loader=FileSystemLoader("."))
NON_ALPHANUM_RE = re.compile(r"[^a-zа-яё0-9]+")

def tokenize(text):
    text = text.lower()
    text = NON_ALPHANUM_RE.sub(" ", text)
    return text.split()


def encode_prompt(example_instructions, settings, template_path):
    template = JINJA_ENV.get_template(template_path)
    for idx, task in enumerate(example_instructions):
        task["instruction"] = re.sub(r"\s+", " ", task["instruction"]).strip().rstrip(":")
        task["input"] = "<noinput>" if not task["input"] else task["input"]
        task["index"] = idx + 1
    return template.render(
        num_tasks=settings["num_tasks"],
        example_tasks=example_instructions
    ).strip() + "\n"


def post_process(response, settings):
    if not response:
        return []
    raw_instructions = response["message"]["content"]
    if raw_instructions.count("###") < 2:
        return []
    raw_instructions = re.split("###", raw_instructions)
    if response["finish_reason"] == "length":
        raw_instructions = raw_instructions[:-1]
    raw_instructions = [i for i in raw_instructions if i.strip()]

    instructions = []
    for idx, fragment in enumerate(raw_instructions):
        final_data = None
        idx = idx + settings["num_example_tasks"] + 1
        for idx_ in (idx, idx - 1, idx + 1):
            special_tokens_re = "(" + "|".join(settings["special_tokens"]) + ")"
            splitted_data = re.split(f"{idx_}\.\s+{special_tokens_re}", fragment)
            if len(splitted_data) == 7:
                final_data = splitted_data
                break

        if not final_data:
            print("Skip fields:", fragment)
            continue

        inst = final_data[2].strip()
        inp = final_data[4].strip()
        inp = "" if "<noinput>" in inp.strip().lower() else inp
        out = final_data[6].strip()

        # filter out too short or too long instructions
        if len(inst.split()) <= 2 or len(inst.split()) > 150:
            print("Skip length:", fragment)
            continue

        # filter based on keywords that are not suitable for language models.
        has_bad_words = False
        for word in settings["blacklist"]:
            if word in inst.lower() or word in inp.lower():
                has_bad_words = True
        if has_bad_words:
            print("Skip blacklist:", fragment)
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            print("Skip punct:", fragment)
            continue

        has_spec_token = False
        for token in settings["special_tokens"]:
            if token in inp or token in out:
                has_spec_token = True
        if has_spec_token:
            print("Skip incorrect parsing:", fragment)
            continue

        instructions.append({"instruction": inst, "input": inp, "output": out})

    return instructions


def generate_instructions(
    output_path,
    seed_tasks_path,
    settings_path,
    template_path,
    num_instructions_to_generate=10000,
    model_name="gpt-3.5-turbo",
    request_batch_size=5,
    temperature=1.0,
    top_p=0.95,
    num_cpus=8,
):
    random.seed(43)
    with open(settings_path) as r:
        settings = json.load(r)

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [{
        "instruction": t["instruction"],
        "input": t["instances"][0]["input"],
        "output": t["instances"][0]["output"]
    } for t in seed_tasks]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    machine_instruction_data = []
    if os.path.exists(output_path):
        with open(output_path) as r:
            machine_instruction_data = json.load(r)
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    all_instructions = [d["instruction"] for d in seed_instruction_data + machine_instruction_data]
    all_instruction_tokens = [tokenize(inst) for inst in all_instructions]

    request_idx = 0
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    is_prompt_printed = False
    is_output_printed = False
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch = []

        for _ in range(request_batch_size):
            prompt_instructions = random.sample(seed_instruction_data, settings["num_example_tasks"] - 1)
            prompt_machine_instructions = random.sample(machine_instruction_data, 1)
            prompt_instructions += prompt_machine_instructions
            random.shuffle(prompt_instructions)

            prompt = encode_prompt(prompt_instructions, settings, template_path)
            messages = [
                {"role": "system", "content": settings["system_message"]},
                {"role": "user", "content": prompt}
            ]
            batch.append(messages)

        if not is_prompt_printed:
            is_prompt_printed = True
            print("Prompt example:")
            for message in batch[0]:
                print("Role: {}, content: {}".format(message["role"], message["content"]))

        request_start = time.time()
        num_tasks =  settings["num_tasks"]
        results = utils.openai_batch_completion(
            batch=batch,
            model_name=model_name,
            decoding_args=utils.OpenAIDecodingArguments(
                temperature=temperature,
                top_p=top_p,
                stop=[f"\n{num_tasks + 1}", "{num_tasks + 1}."]
            )
        )
        if not is_output_printed:
            is_output_printed = True
            print("Output example:")
            print(results[0].message["content"])
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            instruction_data.extend(post_process(result, settings=settings))

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            new_instruction_tokens = tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
                rouge_scores = [score.fmeasure for score in rouge_scores]
            if max(rouge_scores) > 0.7:
                continue

            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }

            keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)

        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        print("===================================")

        with open(output_path + "_tmp", "w") as w:
            json.dump(machine_instruction_data, w, indent=4, ensure_ascii=False)
        shutil.move(output_path + "_tmp", output_path)



def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
