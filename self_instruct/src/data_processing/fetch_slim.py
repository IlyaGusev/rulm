import json
import random

from datasets import load_dataset
import fire


def fetch(train_path, val_path, min_score: int = 8):
    records = []
    for row in load_dataset("IlyaGusev/saiga_scored", split="train"):
        if row["opus_score"] >= min_score:
            records.append(row)
    random.shuffle(records)
    border = int(0.95 * len(records))
    train_records = records[:border]
    val_records = records[border:]
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")



if __name__ == "__main__":
    fire.Fire(fetch)
