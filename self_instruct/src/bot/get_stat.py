from statistics import mean
from collections import Counter, defaultdict
from tinydb import TinyDB, Query

db = TinyDB("db.json", ensure_ascii=False)
print(len(db))

Record = Query()
username_records = list(db.search(Record.username.exists()))
chat2username = {r["chat_id"]: r["username"] for r in username_records if "chat_id" in r and "username" in r}

commiters = [r.get("username") for r in username_records]
commiters += [r.get("chat_id") for r in db.search(Record.chat_id.exists()) if not r.get("username")]
commiters = [c if c not in chat2username else chat2username[c] for c in commiters]
commiters = Counter(commiters)
print(commiters.most_common())

key = ("instruction", "input")

labels = defaultdict(list)
for record in db.all():
    labels[tuple(record[k] for k in key)].append(record["label"])

agg_labels = []
overlaps = []
for key, record_labels in labels.items():
    overlap = len(record_labels)
    overlaps.append(float(overlap))
    agg_label = Counter(record_labels).most_common()[0][0]
    agg_labels.append(agg_label)

print("Avg overlap:", mean(overlaps))
print(Counter(agg_labels).most_common())
