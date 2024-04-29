import json
import random
from itertools import chain

from datasets import load_dataset
import fire

from src.data_processing.bad_substrings import has_bad_ss


def has_empty_bot_messages(messages):
    return sum([len(m["content"].strip()) == 0 for m in messages if m["role"] == "bot"]) >= 1


def fetch_tagengo():
    for row in chain(
        load_dataset("allenai/WildChat", split="train")
    ):
        language = row["language"]
        if language != "Russian":
            continue
        if row["model"] != "gpt-4":
            continue
        messages = row["conversation"]
        messages = [{"content": m["content"], "role": m["role"]} for m in messages]
        if any([m["content"] is None for m in messages]):
            continue
        if has_bad_ss(messages):
            continue
        if has_empty_bot_messages(messages):
            continue
        if messages[1]["content"].lower().startswith("извини"):
            continue
        if len(messages) < 2:
            continue
        yield {
            "messages": messages,
            "source": "wildchat",
            "language": language
        }


def main(output_path):
    with open(output_path, "w") as w:
        for record in fetch_tagengo():
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
