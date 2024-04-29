import sys
import json
import random

import pandas as pd


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


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
        first_message = user_messages[0]["content"]
        words = first_message.split(" ")
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
clean_records = []
for r in records:
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
    assert r["messages"][-1]["content"].strip()
    assert isinstance(r["opus_score"], int)
    assert 1 <= r["opus_score"] <= 10
    assert r["turns"] >= 1
    clean_records.append(r)
records = clean_records
records = undup_by_prefix(records)
records = undup_by_ngrams(records)

random.shuffle(records)

pd.DataFrame(records).to_parquet(output_path)
