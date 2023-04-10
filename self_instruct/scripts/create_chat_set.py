import json
import sys
import random
from datasets import load_dataset
from tqdm import tqdm

train_path = sys.argv[1]
val_path = sys.argv[2]

records = []


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


for row in tqdm(load_dataset("IlyaGusev/ru_turbo_saiga", split="train")):
    row["messages"] = revert_flattening(row["messages"])
    records.append(row)

for row in tqdm(load_dataset("IlyaGusev/ru_sharegpt_cleaned", split="train")):
    row["messages"] = revert_flattening(row["messages"])
    records.append(row)

for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca", split="train")):
    row["output"] = row.pop("alternative_output")
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    row["messages"] = [
        {"role": "user", "content": (row["instruction"] + "\nДано: " + row["input"]) if row["input"] else row["instruction"]},
        {"role": "assistant", "content": row["output"]}
    ]
    if random.random() < 0.5:
        records.append(row)
    else:
        index = random.randrange(len(records))
        records[index]["messages"] += row["messages"]

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
