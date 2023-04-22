import csv
import sys
import json
import random
from collections import defaultdict
from itertools import combinations

from src.util.io import read_jsonl

output_path = sys.argv[1]

files = [
    {
        "file_name": "data/vicuna_saiga7b_answers.jsonl",
        "model_name": "saiga7b",
    },
    {
        "file_name": "data/vicuna_saiga30b_answers.jsonl",
        "model_name": "saiga30b",
    },
    {
        "file_name": "data/user_saiga7b_answers.jsonl",
        "model_name": "saiga7b",
    },
    {
        "file_name": "data/user_saiga30b_answers.jsonl",
        "model_name": "saiga30b",
    }
]
model_count = len({r["model_name"] for r in files})

def get_key(record):
    if "input" not in record:
        return (record["instruction"], )
    return (record["instruction"], record["input"])

answers = defaultdict(list)
for r in files:
    file_name = r["file_name"]
    model_name = r["model_name"]
    records = read_jsonl(file_name)
    for record in records:
        record["model_name"] = model_name
        answers[get_key(record)].append(record)

for key, outputs in answers.items():
    assert len(outputs) == model_count

with open(output_path, "w") as w:
    for key, outputs in answers.items():
        all_answers = [(r["model_name"], r["answer"]) for r in outputs]
        record = {k: v for k, v in outputs[0].items() if k in ("instruction", "input")}
        for comb in combinations(all_answers, 2):
            (a_model, a_answer), (b_model, b_answer) = comb
            if random.random() < 0.5:
                a_model, b_model = b_model, a_model
                a_answer, b_answer = b_answer, a_answer
            record["a"] = a_answer
            record["b"] = b_answer
            record["a_model"] = a_model
            record["b_model"] = b_model
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
