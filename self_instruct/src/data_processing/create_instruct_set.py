import json
import sys
import random
from datasets import load_dataset

train_path = sys.argv[1]
val_path = sys.argv[2]

records = []


for row in load_dataset("IlyaGusev/ru_turbo_alpaca", split="train"):
    row["output"] = row.pop("alternative_output")
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
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
