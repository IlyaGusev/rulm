import csv
import sys
import json
import random

from src.util.io import read_jsonl

input1 = sys.argv[1]
input2 = sys.argv[2]
model1_id = sys.argv[3]
model2_id = sys.argv[4]
output_path = sys.argv[5]

if input1.endswith("csv"):
    records1 = []
    with open(input1) as r:
        reader = csv.reader(r)
        header = next(reader)
        for row in reader:
            record = dict(zip(header, row))
            record["instruction"] = record.pop("Instruction")
            record["answer"] = record.pop("Ans RetrivalR").split("<instructionS>")[-1].replace("<instructionE>", "").strip()
            records1.append(record)
elif input1.endswith("jsonl"):
    records1 = read_jsonl(input1)

if input2.endswith("csv"):
    records2 = []
    with open(input2) as r:
        reader = csv.reader(r)
        header = next(reader)
        for row in reader:
            record = dict(zip(header, row))
            record["instruction"] = record.pop("Instruction")
            record["answer"] = record.pop("Ans ClassicR").split("<instructionS>")[-1].replace("<instructionE>", "").strip()
            records2.append(record)
elif input2.endswith("jsonl"):
    records2 = read_jsonl(input2)


with open(output_path, "w") as w:
    for r1, r2 in zip(records1, records2):
        assert r1["instruction"] == r2["instruction"]
        a_answer = r1["answer"].replace("assistant", "").strip()
        b_answer = r2["answer"].replace("assistant", "").strip()
        record = {k: v for k, v in r1.items() if k in ("instruction", "input")}
        a_model, b_model = model1_id, model2_id
        if random.random() < 0.5:
            a_model, b_model = b_model, a_model
            a_answer, b_answer = b_answer, a_answer
        record["a"] = a_answer
        record["b"] = b_answer
        record["a_model"] = a_model
        record["b_model"] = b_model
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
