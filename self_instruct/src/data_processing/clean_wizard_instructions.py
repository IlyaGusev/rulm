import json
import sys

from src.data_processing.bad_substrings import has_bad_ss

input_path = sys.argv[1]
output_path = sys.argv[2]


with open(input_path) as r, open(output_path, "w") as w:
    for line in r:
        record = json.loads(line)
        record.pop("input")
        instruction = record["instruction"]
        output = record["output"]
        if "ИИ" in instruction:
            continue
        if has_bad_ss([{"content": output}]):
            continue
        if not output.strip():
            continue
        lines = instruction.split("\n")
        if "Ответ:" in lines[-1]:
            continue
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
