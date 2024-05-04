import random

import fire
import pandas as pd

from datasets import load_dataset

from src.util.io import read_jsonl

def to_preferences(input_path, output_path):
    records = read_jsonl(input_path)
    new_records = []
    for record in records:
        messages = record["messages"]
        answer = record["answer"]
        if messages[-1]["role"] not in ("assistant", "bot"):
            continue
        prompt = messages[:-1]
        chosen = [messages[-1]]
        rejected = [{"role": "assistant", "content": answer}]
        new_records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "gpt4_vs_saiga"
        })

    for row in load_dataset("IlyaGusev/saiga_reward", split="train"):
        new_records.append(row)
    random.shuffle(new_records)
    pd.DataFrame(new_records).to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(to_preferences)
