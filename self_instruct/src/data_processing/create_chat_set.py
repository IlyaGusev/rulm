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

max_length = max([sum([len(m["content"]) for m in r["messages"]]) for r in records])
print("Max Saiga length:", max_length)

alpaca_records = []
for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca", split="train")):
    row["output"] = row.pop("alternative_output")
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    row["messages"] = [
        {"role": "user", "content": (row["instruction"] + "\nДано: " + row["input"]) if row["input"] else row["instruction"]},
        {"role": "bot", "content": row["output"]}
    ]
    alpaca_records.append(row)

merged_alpaca_records = []
prev_record_idx = None
print("Before merge:", len(alpaca_records))
for idx, record in enumerate(alpaca_records):
    text_length = sum([len(m["content"]) for m in record["messages"]])
    if text_length > 1000:
        merged_alpaca_records.append(record)
        continue
    if prev_record_idx is None:
        prev_record_idx = idx
        continue
    messages = alpaca_records[prev_record_idx]["messages"] + record["messages"]
    merged_alpaca_records.append({
        "messages": messages
    })
    prev_record_idx = None
print("After merge:", len(merged_alpaca_records))
alpaca_records = merged_alpaca_records

max_length = max([sum([len(m["content"]) for m in r["messages"]]) for r in alpaca_records])
print("Max Alpaca length:", max_length)

excluded_indices = set()
for record in tqdm(alpaca_records):
    text_length = sum([len(m["content"]) for m in record["messages"]])
    if text_length > 1500:
        records.append(record)
        continue
    if random.random() < 0.5:
        records.append(record)
        continue
    index = random.randrange(len(records))
    while index in excluded_indices:
        index = random.randrange(len(records))
    excluded_indices.add(index)
    records[index]["messages"] += record["messages"]

for row in tqdm(load_dataset("IlyaGusev/ru_sharegpt_cleaned", split="train")):
    row["messages"] = revert_flattening(row["messages"])
    text_length = sum([len(m["content"]) for m in row["messages"]])
    if text_length > 6000:
        continue
    records.append(row)

max_length = max([sum([len(m["content"]) for m in r["messages"]]) for r in records])
print("Max length:", max_length)

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
