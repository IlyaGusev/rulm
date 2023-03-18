import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import fire
import numpy as np
import tqdm
from rouge_score import rouge_scorer

import utils


def encode_prompt(prompt_instructions, prompt_path):
    with open(prompt_path) as r:
        messages = [json.loads(line) for line in r]
        prompt = messages[-1]["content"]
    for idx, task_dict in enumerate(prompt_instructions):
        instruction, inp, out = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        inp = "<noinput>" if not inp else inp
        prompt += f"\n###\n"
        prompt += f"{idx + 1}. Задание: {instruction}\n"
        prompt += f"{idx + 1}. Вход:\n{inp}\n"
        prompt += f"{idx + 1}. Выход:\n{out}\n"
    messages[-1]["content"] = prompt
    return messages


def post_process(response, num_prompt_instructions, blacklist_path):
    if not response:
        return []

    raw_instructions = response["message"]["content"]
    if raw_instructions.count("###") > 2:
        raw_instructions = re.split("###", raw_instructions)
    else:
        raw_instructions = raw_instructions.split("\n\n")

    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue

        final_data = None
        idx += num_prompt_instructions + 1
        for idx_ in (idx, idx - 1, idx + 1):
            splitted_data = re.split(f"{idx_}\.\s+(Задание|Вход|Выход):", inst)
            if len(splitted_data) == 7:
                final_data = splitted_data
                break
        if not final_data:
            print("Skip fields:", inst)
            continue

        inst = final_data[2].strip()
        inp = final_data[4].strip()
        inp = "" if "<noinput>" in inp.strip().lower() else inp
        out = final_data[6].strip()

        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            print("Skip length:", inst)
            continue

        # filter based on keywords that are not suitable for language models.
        has_bad_words = False
        with open(blacklist_path) as r:
            blacklist = [l.strip() for l in r.readlines()]
            for word in blacklist:
                if word in inst.lower() or word in inp.lower():
                    has_bad_words = True
        if has_bad_words:
            print("Skip blacklist:", inst + inp)
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            print("Skip punct:", inst)
            continue

        instructions.append({"instruction": inst, "input": inp, "output": out})

    return instructions


def generate_instructions(
    output_path,
    seed_tasks_path,
    prompt_path,
    blacklist_path,
    num_instructions_to_generate=50,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=4,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=8,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["instances"][0]["input"],
            "output": t["instances"][0]["output"]
        } for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    machine_instruction_data = []
    if os.path.exists(output_path):
        with open(output_path) as r:
            machine_instruction_data = json.load(r)
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    all_instructions = [d["instruction"] for d in seed_instruction_data + machine_instruction_data]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    request_idx = 0
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    is_prompt_printed = False
    is_output_printed = False
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        messages = encode_prompt(prompt_instructions, prompt_path)
        if not is_prompt_printed:
            is_prompt_printed = True
            print("Prompt example:")
            for message in messages:
                print("Role: {}, content: {}".format(message["role"], message["content"]))

        request_start = time.time()
        result = utils.openai_completion(
            messages=messages,
            model_name=model_name,
            decoding_args=utils.OpenAIDecodingArguments(
                temperature=temperature,
                top_p=top_p,
                stop=["\n11", "11.", "11."]
            )
        )
        if not is_output_printed:
            is_output_printed = True
            print("Output example:")
            print(result.message["content"])
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = post_process(
            result,
            num_prompt_instructions=num_prompt_instructions,
            blacklist_path=blacklist_path
        )

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue

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

        with open(output_path, "w") as w:
            json.dump(machine_instruction_data, w, indent=4, ensure_ascii=False)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
