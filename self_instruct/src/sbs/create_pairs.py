import json
import random
from collections import defaultdict
from itertools import combinations

import fire

from src.util.io import read_jsonl


def get_key(record):
    if "input" not in record:
        return (record["instruction"], )
    return (record["instruction"], record["input"])


def create_pairs(config_path, output_path):
    with open(config_path) as r:
        config = json.load(r)
    files = config["files"]
    pairs_to_compare = config["pairs_to_compare"]
    pairs_to_compare = [tuple(p) for p in pairs_to_compare]
    model_count = len({r["model_name"] for r in files})

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


if __name__ == "__main__":
    fire.Fire(create_pairs)
