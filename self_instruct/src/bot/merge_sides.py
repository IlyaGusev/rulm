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
        "file_name": "data/vicuna_saiga30b_answers.jsonl",
        "model_name": "saiga30b",
    },
    {
        "file_name": "data/user_v2_saiga30b_answers.jsonl",
        "model_name": "saiga30b",
    },
    {
        "file_name": "data/vicuna_saiga30bq4_1_answers.jsonl",
        "model_name": "saiga30bq4_1",
    },
    {
        "file_name": "data/user_v2_saiga30bq4_1_answers.jsonl",
        "model_name": "saiga30bq4_1",
    },
    {
        "file_name": "data/vicuna_turbo_answers.jsonl",
        "model_name": "turbo"
    },
    {
        "file_name": "data/user_v2_turbo_answers.jsonl",
        "model_name": "turbo"
    },
    {
        "file_name": "data/vicuna_gpt4_answers.jsonl",
        "model_name": "gpt4"
    },
    {
        "file_name": "data/user_v2_gpt4_answers.jsonl",
        "model_name": "gpt4"
    },

]

pairs_to_compare = [
    ("turbo", "saiga30bq4_1"),
    ("turbo", "saiga30b"),
    ("turbo", "gpt4"),
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
        if "input" not in record:
            record["input"] = None
        if "output" in record:
            record["answer"] = record.pop("output")
        record["instruction"] = record["instruction"].strip()
        record["model_name"] = model_name
        answers[get_key(record)].append(record)

for key, outputs in answers.items():
    print(outputs)
    assert len(outputs) == model_count

num_samples = len(answers)
print(len(answers))

with open(output_path, "w") as w:
    for idx1, (key, outputs) in enumerate(answers.items()):
        all_answers = [(r["model_name"], r["answer"]) for r in outputs]
        record = {k: v for k, v in outputs[0].items() if k in ("instruction", "input")}
        for idx2, comb in enumerate(combinations(all_answers, 2)):
            (a_model, a_answer), (b_model, b_answer) = comb
            if (a_model, b_model) not in pairs_to_compare and (b_model, a_model) not in pairs_to_compare:
                continue
            if random.random() < 0.5:
                a_model, b_model = b_model, a_model
                a_answer, b_answer = b_answer, a_answer
            record["id"] = "task_{}_{}".format(idx1, idx2)
            record["left_answer"] = a_answer
            record["right_answer"] = b_answer
            record["left_model"] = a_model
            record["right_model"] = b_model
            if not record["input"]:
                record["input"] = ""
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
