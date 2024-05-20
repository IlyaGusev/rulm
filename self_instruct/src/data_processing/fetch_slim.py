import json
import random

import mmh3
from datasets import load_dataset
import fire


def fetch(train_path, val_path, min_score: int = 9):
    records = []
    for row in load_dataset("IlyaGusev/saiga_scored", split="train"):
        score = row["opus_score"]
        is_bad_by_regex = row["is_bad_by_regex"]
        if score < min_score or is_bad_by_regex:
            continue
        records.append(row)

    random.shuffle(records)

    train_records = []
    val_records = []
    for r in records:
        s = str(r["messages"])
        h = mmh3.hash(s, signed=False)
        if h % 100 < 97:
            train_records.append(r)
        else:
            val_records.append(r)
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")



if __name__ == "__main__":
    fire.Fire(fetch)
