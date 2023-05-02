import json


def read_jsonl(file_name):
    with open(file_name) as r:
        return [json.loads(line) for line in r]


def write_jsonl(records, path):
    with open(path, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


