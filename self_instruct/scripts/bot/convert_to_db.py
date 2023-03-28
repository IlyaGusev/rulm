import json
import sys
from tinydb import TinyDB

input_path = sys.argv[1]
db_path = sys.argv[2]
db = TinyDB(db_path, ensure_ascii=False)

with open(input_path) as r:
    for line in r:
        record = json.loads(line)
        db.insert(record)
