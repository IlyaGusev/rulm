import json
import sys
import re
import random
from datasets import load_dataset
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH, LeanMinHash


BAD_SS = (
    " ул. ",
    " +7",
    "как ии",
    "как ai",
    "как аи",
    "как модель ии",
    "как алгоритм",
    "языковая модель ии",
    "как искусственный интеллект",
    "как нейросеть",
    "виртуальный ассистент",
    "виртуальный помощник",
    "как нейронная сеть",
    "онлайн-ассистент",
    "интеллектуальный помощник",
    "голосовой помощник",
    "искусственный разум",
    "компьютерная программа",
    "программный помощник",
    "представитель ии",
    "ассистент ии",
    "ии-ассистент",
    "умный искусственный интеллект",
    "помощник ai",
    "как ассистент",
    "как помощник",
    "как иси-ассистент"
    "ai помощник",
    "я - искусственный интеллект",
    "я являюсь искусственным интеллектом",
    "я искусственный интеллект",
    "я – искусственный интеллект",
    "я - искуственный интеллект",
    "в качестве ии",
    "в качестве искуственного интеллекта",
    "от лица ии",
    "от лица искуственного интеллекта",
    "openai",
    "chatgpt",
    "as a language model",
    "as an ai",
    "к сожалению",
    "sorry",
    "я - алгоритм",
    "я – алгоритм",
    "я - компьютерная программа",
    "я – компьютерная программа",
    "я компьютерная программа",
    "я являюсь компьютерной программой",
    "я - ai",
    "я – ai",
    "я ai",
    "я являюсь ai",
    "я - ии",
    "я – ии",
    "я ии",
    "я являюсь ии",
    "я - виртуальный помощник",
    "я – виртуальный помощник",
    "я виртуальный помощник",
    "я являюсь виртуальным помощником",
    "я - виртуальный ассистент",
    "я – виртуальный ассистент",
    "я виртуальный ассистент",
    "я являюсь виртуальным ассистентом",
    "я - программа",
    "я – программа",
    "я программа",
    "я являюсь программой",
    "я - ассистент",
    "я – ассистент",
    "я ассистент"
)

def has_bad_ss(messages):
    for m in messages:
        text = m["content"].lower()
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


def build_char_system_messages(char):
    name = char["name"]
    context = char["context"]
    greeting = char["greeting"]
    example_dialogue = char["example_dialogue"]

    context = f"Ты {name}. {context}"
    chat = []
    if random.random() < 0.2:
        context += f"\nПриветствие: {greeting}"
        chat.append({
            "role": "bot",
            "content": greeting
        })
    if random.random() < 0.2:
        mapping = {
            "user": "Пользователь",
            "char": "Персонаж"
        }
        example_messages = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue]
        context += "\nПример диалога:\n" + "\n".join(example_messages)
    chat.insert(0, {
        "role": "system",
        "content": context
    })
    return chat


def re_tokenize(text):
    return re.findall(r'[а-яё-]+|[a-z-]+|\d+|\S', text, re.I)


def ngrams(sequence, n):
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def calc_fingerprint(text, ngram_size: int = 1, num_perm: int = 128):
    tokens = re_tokenize(text)
    if ngram_size > 1:
        tokens = {" ".join(t) for t in ngrams(tokens, ngram_size)}
    tokens = [token.encode('utf-8') for token in tokens]

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(tokens)

    lean_minhash = LeanMinHash(minhash)
    buf = bytearray(lean_minhash.bytesize())
    lean_minhash.serialize(buf)

    return buf


def undup_alpaca(alpaca_records, num_perm: int = 32):
    for record in tqdm(alpaca_records, desc="Fingerprinting"):
        record["minhash"] = calc_fingerprint(record["messages"][0]["content"], num_perm=num_perm)

    threshold = 0.6
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm
    )

    filtered_records = []
    for idx, record in tqdm(enumerate(alpaca_records), desc="Undup"):
        minhash = LeanMinHash.deserialize(record["minhash"])
        is_dup = False
        for other_idx in lsh.query(minhash):
            other_record = alpaca_records[other_idx]
            other_minhash = LeanMinHash.deserialize(other_record["minhash"])
            if minhash.jaccard(other_minhash) > threshold:
                is_dup = True
        if is_dup:
            continue
        lsh.insert(idx, minhash)
        filtered_records.append(record)
    for record in filtered_records:
        record.pop("minhash")
    return filtered_records


def main(train_path, val_path):
    random.seed(42)
    records = []

    alpaca_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca_evol_instruct", split="train")):
        instruction = row["instruction"]
        output = row["output"]
        if has_bad_ss([{"content": output}]):
            continue
        alpaca_records.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "bot", "content": output}
            ],
            "source": "alpaca-evol-instruct"
        })
    print("Alpaca EI count:", len(alpaca_records))
    print("Max Alpaca EI length:", calc_max_length(alpaca_records))

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

    alpaca_records = undup_alpaca(alpaca_records)
    print("Alpaca after undup count:", len(alpaca_records))

    for row in tqdm(load_dataset("IlyaGusev/gpt_roleplay_realm", split="ru")):
        name = row["name"]
        context = row["context"]
        greeting = row["greeting"]
        example_dialogue = row["example_dialogue"]
        for dialogue in row["dialogues"]:
            chat = dialogue["chat"]
            for message in chat:
                if message["role"] == "char":
                    message["role"] = "bot"
                if message["role"] == "operator":
                    message["role"] = "user"

            system_messages = build_char_system_messages(row)
            chat = system_messages + chat
            records.append({
                "messages": chat,
                "source": "roleplay"
            })

    print("Roleplay count:", len(records))

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
            assert role in ("bot", "user", "system"), role
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
