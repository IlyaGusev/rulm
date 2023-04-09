import sys
import json
import random

from datasets import load_dataset

random.seed(42)
output_path = sys.argv[1]

seeds = list()
for row in load_dataset("IlyaGusev/ru_stackoverflow", split="train"):
    if random.random() < 0.045:
        seeds.append({
            "seed": row["title"],
            "source": "ru_stackoverflow"
        })

for row in load_dataset("its5Q/habr_qna", split="train"):
    if random.random() < 0.025:
        seeds.append({
            "seed": row["title"],
            "source": "habr_qna"
        })

for row in load_dataset("its5Q/yandex-q", split="train"):
    if random.random() < 0.09:
        seeds.append({
            "seed": row["question"],
            "source": "yandex_q"
        })



filtered_seeds = []
seeds_set = set()
for record in seeds:
    if record["seed"] in seeds_set:
        continue
    if len(record["seed"]) < 10:
        continue
    if len(record["seed"]) > 300:
        print(record["seed"])
        continue
    seeds_set.add(record["seed"])
    filtered_seeds.append(record)
seeds = filtered_seeds
random.shuffle(seeds)


with open(output_path, "w") as w:
    for record in seeds:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
