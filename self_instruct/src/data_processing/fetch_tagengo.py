import json
import random
from itertools import chain

from datasets import load_dataset
import fire

from src.data_processing.bad_substrings import has_bad_ss


def has_empty_bot_messages(messages):
    return sum([len(m["content"].strip()) == 0 for m in messages if m["role"] == "bot"]) >= 1


def fetch_tagengo():
    mapping = {
        "gpt": "bot",
        "human": "user"
    }
    for row in chain(
        load_dataset("lightblue/tagengo-gpt4", split="train"),
        load_dataset("lightblue/gpt4_conversations_multilingual", split="train")
    ):
        language = row["language"]
        if language not in ("Russian", None):
            continue
        if language != "Russian" and random.random() > 0.1:
            continue
        if language is None:
            language = "Mixed"
        messages = row["conversations"]
        messages = [{"content": m["value"], "role": mapping[m["from"]]} for m in messages]
        if any([m["content"] is None for m in messages]):
            continue
        if "response" in row and row["response"][1] != "stop":
            continue
        if has_bad_ss(messages):
            continue
        if has_empty_bot_messages(messages):
            continue
        if messages[-1]["content"].lower().startswith("извини"):
            continue
        yield {
            "messages": messages,
            "source": "tagengo",
            "language": language
        }


def main(output_path):
    with open(output_path, "w") as w:
        for record in fetch_tagengo():
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
