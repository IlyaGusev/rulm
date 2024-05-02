import random
import json

import pandas as pd
import fire

from datasets import load_dataset

def to_reward_dataset(input_path, tasks_path, output_path):
    with open(input_path) as r:
        records = json.load(r)
    with open(tasks_path) as r:
        tasks = [json.loads(line) for line in r]
        instructions = {t["instruction"] for t in tasks}
    new_records = []
    existing_keys = set()
    for r in records:
        preference = r["preference"]
        output_1 = r["output_1"]
        output_2 = r["output_2"]
        if output_1 == output_2:
            continue
        prompt = r["instruction"]
        if prompt not in instructions:
            continue
        if prompt.startswith("###"):
            continue
        winning_output = output_1 if preference < 1.5 else output_2
        losing_output = output_2 if preference < 1.5 else output_1
        key = (prompt, winning_output, losing_output)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_records.append({
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": winning_output}],
            "rejected": [{"role": "assistant", "content": losing_output}],
            "source": "saiga_tasks"
        })
    for row in load_dataset("tasksource/oasst2_pairwise_rlhf_reward", split="train"):
        if row["lang"] != "ru":
            continue
        prompt = row["prompt"]
        prompt_messages = []
        for line in prompt.split("\n"):
            if line.startswith("prompter:"):
                line = line.replace("prompter:", "").lstrip()
                prompt_messages.append({"role": "user", "content": line})
            elif line.startswith("assistant:"):
                line = line.replace("assistant:", "").lstrip()
                prompt_messages.append({"role": "assistant", "content": line})
            else:
                prompt_messages[-1]["content"] += "\n" + line


        chosen = row["chosen"]
        rejected = row["rejected"]
        new_records.append({
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "source": "oasst"
        })

    random.shuffle(new_records)
    pd.DataFrame(new_records).to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(to_reward_dataset)
