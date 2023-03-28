import sys
import json

orig_path = sys.argv[1]
markup_path = sys.argv[2]
output_path = sys.argv[3]
new_path = sys.argv[4]


def get_key(r):
    inp = r["input"].strip()
    if "noinput" in inp:
        inp = ""
    return tuple((r["instruction"].strip(), inp))

with open(orig_path) as r:
    orig_records = {get_key(r): r for r in json.load(r)}
print(len(orig_records))

count = 0
with open(markup_path) as r:
    for line in r:
        record = json.loads(line)
        key = get_key(record)
        if key in orig_records:
            orig_records[key] = record
            count += 1
print(count)

with open(new_path) as r:
    for line in r:
        record = json.loads(line)
        key = get_key(record)
        if key in orig_records:
            orig_records[key]["alternative_output"] = record["new_output"]

mapping = {
    "all_ok": "ok",
    "ok": "bad_output",
    "bad": "bad_task"
}

with open(output_path, "w") as w:
    for _, record in orig_records.items():
        record.pop("most_similar_instructions", None)
        record.pop("avg_similarity_score", None)
        record.pop("index", None)
        if "alternative_output" not in record:
            continue
        if "noinput" in record["input"].strip():
            record["input"] = ""
        if "label" not in record:
            record["label"] = None
            record["all_labels"] = []
        else:
            record["label"] = mapping[record["label"]]
            record["all_labels"] = [mapping[l] for l in record["all_labels"]]
        if "agreement" not in record:
            record["agreement"] = None
        if "overlap" not in record:
            record["overlap"] = None
        if len(record["output"]) <= 1 or len(record["alternative_output"]) <= 1:
            continue
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
