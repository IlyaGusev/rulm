import json
import sys

from datasets import load_dataset

output_path = sys.argv[1]

with open(output_path, "w") as w:
    gazeta = load_dataset('IlyaGusev/gazeta', revision="v2.0")["train"]
    for row in gazeta:
        record = {
            "text": row["text"],
            "source": "gazeta",
            "meta": {
                "title": row["title"],
                "date": row["date"],
                "url": row["url"]
            }
        }
        print(record)
        w.write(json.dumps(record, ensure_ascii=False) + "\n")
        break

    medical_qa = load_dataset("blinoff/medical_qa_ru_data")["train"]
    for row in medical_qa:
        record = {
            "text": row["desc"],
            "source": "medical_qa",
            "meta": {
                "theme": row["theme"],
                "date": row["date"],
                "categ": row["categ"],
                "spec10": row["spec10"]
            }
        }
        print(record)
        w.write(json.dumps(record, ensure_ascii=False) + "\n")
        for ans in row["ans"].split(";\n"):
            record = {
                "text": ans,
                "source": "medical_qa",
                "meta": {
                    "theme": row["theme"],
                    "date": row["date"],
                    "categ": row["categ"],
                    "spec10": row["spec10"]
                }
            }
            print(record)
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
        break

    #cc100 = load_dataset("cc100", lang="ru")
    #for row in cc100:
    #    print(row)
    #    break
