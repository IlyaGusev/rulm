import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
with open(input_path) as r, open(output_path, "w") as w:
    data = json.load(r)
    for record in data:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
