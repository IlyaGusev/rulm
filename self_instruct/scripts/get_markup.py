import sys
import json
from statistics import mean
from collections import Counter, defaultdict
from tinydb import TinyDB, Query

output_path = sys.argv[1]

db = TinyDB("db.json", ensure_ascii=False)

key = ("instruction", "input")

records = dict()
labels = defaultdict(list)
for record in db.all():
    r_key = tuple(record[k] for k in key)
    labels[r_key].append(record["label"])
    records[r_key] = record

agreements = []
agg_labels = dict()
for key, record_labels in labels.items():
    agg_label, agg_label_count = Counter(record_labels).most_common()[0]
    overlap = len(record_labels)

    records[key]["all_labels"] = record_labels
    records[key]["label"] = agg_label
    records[key].pop("username", None)
    records[key].pop("chat_id", None)
    records[key]["overlap"] = overlap
    records[key]["agreement"] = float(agg_label_count) / overlap
    if overlap >= 2:
        agreements.append(records[key]["agreement"])

print(mean(agreements), len(agreements))

with open(output_path, "w") as w:
    for _, record in records.items():
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
