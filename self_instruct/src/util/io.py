import json


def read_jsonl(file_name):
    with open(file_name) as r:
        return [json.loads(line) for line in r]
