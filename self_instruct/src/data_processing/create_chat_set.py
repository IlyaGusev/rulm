import json
import sys
import random
from datasets import load_dataset
from tqdm import tqdm


BAD_SS = (
    " ул. ",
    " +7",
    "Как ИИ",
    "как ИИ",
    "Как модель ИИ",
    "как модель ИИ",
    "как языковая модель ИИ",
    "Как языковая модель ИИ",
    "как искусственный интеллект",
    "Как искусственный интеллект",
    "Я - искусственный интеллект",
    "я - искусственный интеллект",
    "Я являюсь искусственным интеллектом",
    "я являюсь искусственным интеллектом",
    "я искусственный интеллект",
    "OpenAI",
    "ChatGPT",
    "OpenAssistant",
    "Ася",
    "as a language model"
)

def has_bad_ss(messages):
    for m in messages:
        text = m["content"]
        if any(ss in text for ss in BAD_SS):
            return True
    return False


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def main(train_path, val_path):
    records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_saiga", split="train")):
        messages = revert_flattening(row["messages"])
        if has_bad_ss(messages):
            continue
        records.append({
            "messages": messages,
            "source": "saiga"
        })
    print("Saiga count:", len(records))
    print("Max Saiga length:", calc_max_length(records))

    alpaca_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca", split="train")):
        message = row["instruction"]
        if row["input"]:
            message += "\nДано: " + row["input"]
        output = row["alternative_output"]
        if has_bad_ss([{"content": output}]):
            output = row["output"]
            if has_bad_ss([{"content": output}]):
                continue
        alpaca_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": "alpaca"
        })
    print("Alpaca count:", len(alpaca_records))
    print("Max Alpaca length:", calc_max_length(alpaca_records))

    merged_alpaca_records = []
    prev_record_idx = None
    for idx, record in enumerate(alpaca_records):
        text_length = sum([len(m["content"]) for m in record["messages"]])
        if text_length > 1000:
            merged_alpaca_records.append(record)
            continue
        if prev_record_idx is None:
            prev_record_idx = idx
            continue
        messages = alpaca_records[prev_record_idx]["messages"] + record["messages"]
        merged_alpaca_records.append({
            "messages": messages,
            "source": "merged_alpaca"
        })
        prev_record_idx = None
    print("Merged Alpaca count:", len(merged_alpaca_records))
    print("Max Merged Alpaca length:", calc_max_length(alpaca_records))
    alpaca_records = merged_alpaca_records

    excluded_indices = set()
    for record in tqdm(alpaca_records):
        text_length = sum([len(m["content"]) for m in record["messages"]])
        if text_length > 1500 or random.random() < 0.5:
            records.append(record)
            continue
        index = random.randrange(len(records))
        while index in excluded_indices:
            index = random.randrange(len(records))
        excluded_indices.add(index)
        records[index]["source"] = "mixed"
        records[index]["messages"] += record["messages"]
    print("Saiga + Alpaca count:", len(records))
    print("Max Saiga + Alpaca length:", calc_max_length(records))

    for row in tqdm(load_dataset("IlyaGusev/ru_sharegpt_cleaned", split="train")):
        messages = revert_flattening(row["messages"])
        text_length = sum([len(m["content"]) for m in messages])
        while text_length > 10000 and messages:
            messages = messages[:-2]
            text_length = sum([len(m["content"]) for m in messages])
        if not messages:
            continue
        records.append({
            "messages": messages,
            "source": "sharegpt"
        })
    print("Saiga + Alpaca + ShareGPT count:", len(records))
    print("Saiga + Alpaca + ShareGPT max length:", calc_max_length(records))

    for row in tqdm(load_dataset("IlyaGusev/oasst1_ru_main_branch", split="train")):
        messages = revert_flattening(row["messages"])
        text_length = sum([len(m["content"]) for m in messages])
        while text_length > 10000 and messages:
            messages = messages[:-2]
            text_length = sum([len(m["content"]) for m in messages])
        if not messages:
            continue
        records.append({
            "messages": messages,
            "source": "oasst"
        })
    print("All count:", len(records))
    print("All max length:", calc_max_length(records))

    cleaned_records = []
    for record in records:
        messages = record["messages"]
        roles = {m["role"] for m in messages}
        for role in roles:
            assert role in ("bot", "user")
        if has_bad_ss(messages):
            continue
        if not record["messages"]:
            continue
        cleaned_records.append(record)
    records = cleaned_records
    print("All count after cleaning:", len(records))


    random.shuffle(records)
    border = int(0.95 * len(records))
    train_records = records[:border]
    val_records = records[border:]
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


train_path = sys.argv[1]
val_path = sys.argv[2]
main(train_path, val_path)
