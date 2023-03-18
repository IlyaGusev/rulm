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
        prompt = r.read() + "\n"
    for idx, task_dict in enumerate(prompt_instructions):
        instruction, inp, out = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        inp = "<noinput>" if not inp else inp
        prompt += f"###\n"
        prompt += f"{idx + 1}. Задание: {instruction}\n"
        prompt += f"{idx + 1}. Вход:\n{inp}\n"
        prompt += f"{idx + 1}. Выход:\n{out}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Задание:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response, blacklist_path):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Задание:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Задание|Вход|Выход):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            inp = splitted_data[4].strip()
            inp = "" if inp.lower() == "<noinput>" else inp
            out = splitted_data[6].strip()

        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue

        # filter based on keywords that are not suitable for language models.
        with open(blacklist_path) as r:
            blacklist = [l.strip() for l in r.readlines()]
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue

        instructions.append({"instruction": inst, "input": inp, "output": out})

    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instructions(
    output_path,
    seed_tasks_path,
    prompt_path,
    blacklist_path,
    num_instructions_to_generate=10,
    model_name="text-davinci-003",
    num_prompt_instructions=2,
    request_batch_size=5,
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
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, prompt_path)
            batch_inputs.append(prompt)

        request_start = time.time()
        #print(batch_inputs[0])
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=utils.OpenAIDecodingArguments(
                temperature=temperature,
                n=1, max_tokens=1024,
                top_p=top_p, stop=["\n5", "5.", "5."]
            ),
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        print(results)
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result, blacklist_path)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
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
