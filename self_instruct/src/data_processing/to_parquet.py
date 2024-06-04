import sys
import json
import random

from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
from src.data_processing.clean_user_conversations import set_regex_flag


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        records = []
        for idx, line in enumerate(r):
            records.append(json.loads(line))
    return records


def undup_by_prefix(records, prefix_len: int = 40):
    records.sort(key=lambda x: x["opus_score"])
    new_records = {}
    for r in records:
        user_messages = [m for m in r["messages"] if m["role"] == "user"]
        if not user_messages:
            continue
        first_message = user_messages[0]["content"][:prefix_len]
        new_records[first_message] = r
    new_records = list(new_records.values())
    print(len(records), len(new_records))
    return new_records


def generate_ngrams(elements, n: int):
    return {tuple(elements[i:i+n]) for i in range(len(elements) - n + 1)}


def undup_by_ngrams(records, n: int = 8):
    existing_ngrams = dict()
    new_records = []
    records.sort(key=lambda x: x["opus_score"])
    for r in records:
        user_messages = [m for m in r["messages"] if m["role"] == "user"]
        if not user_messages:
            continue
        first_messages = [m["content"] for m in user_messages[:2]]
        words = []
        for m in first_messages:
            words.extend(m.split())
        n_grams = generate_ngrams(words, n)
        skip = False
        for n_gram in n_grams:
            if n_gram in existing_ngrams:
                skip = True
            existing_ngrams[n_gram] = r
        if skip:
            continue
        new_records.append(r)
    print(len(records), len(new_records))
    return new_records


input_path = sys.argv[1]
output_path = sys.argv[2]

records = read_jsonl(input_path)
print(len(records))
clean_records = []
for row in load_dataset("IlyaGusev/saiga_scored", split="train"):
    clean_records.append(row)
print(len(clean_records))
for r in tqdm(records):
    for m in r["messages"]:
        if m["role"] == "assistant":
            m["role"] = "bot"
    r["opus_score"] = int(r.pop("score"))
    if "language" not in r:
        r["language"] = "Russian"
    roles = {m["role"] for m in r["messages"]}
    if "user" not in roles or "bot" not in roles:
        continue
    r["turns"] = sum([m["role"] == "bot" for m in r["messages"]])
    if r["messages"][-1]["role"] != "bot":
        r["messages"] = r["messages"][:-1]
    assert r["messages"][-1]["role"] == "bot"
    if not r["messages"][-1]["content"].strip():
        continue
    assert isinstance(r["opus_score"], int)
    assert 1 <= r["opus_score"] <= 10
    assert r["turns"] >= 1
    topics = r.pop("topics_answer")
    r["sonnet_topic"] = topics["topic"]
    r["sonnet_topic_explanation"] = topics["topic_explanation"]
    r["sonnet_complexity"] = topics["complexity"]
    r["sonnet_complexity_explanation"] = topics["complexity_explanation"]
    clean_records.append(r)
print(len(clean_records))
records = clean_records
records = set_regex_flag(records)
print(sum([r["is_bad_by_regex"] is False for r in records]))
print(len(records))
#records = undup_by_prefix(records)
records = undup_by_ngrams(records)
print(len(records))

random.shuffle(records)

pd.DataFrame(records).to_parquet(output_path)
