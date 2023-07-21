import random
import fire
from typing import List
from itertools import chain

from datasets import load_dataset

from src.eval_rsg import (
    RWSD_PROMPT,
    TERRA_PROMPT,
    MUSERC_SINGLE_PROMPT,
    PARUS_CAUSE_PROMPT,
    PARUS_EFFECT_PROMPT,
    RCB_PROMPT,
    RUCOS_PROMPT,
    RUCOS_MASK,
    rucos_clean_text
)
from src.util.io import write_jsonl

HF_DATASET = "RussianNLP/russian_super_glue"


MUSERC_SOURCE_TEMPLATE = MUSERC_SINGLE_PROMPT
PARUS_CAUSE_SOURCE_TEMPLATE = PARUS_CAUSE_PROMPT
PARUS_EFFECT_SOURCE_TEMPLATE = PARUS_EFFECT_PROMPT
RCB_SOURCE_TEMPLATE = RCB_PROMPT
RUCOS_SOURCE_TEMPLATE = RUCOS_PROMPT
RWSD_SOURCE_TEMPLATE = RWSD_PROMPT
TERRA_SOURCE_TEMPLATE = TERRA_PROMPT

#RWSD_SOURCE_TEMPLATE = 'Текст: "{text}"\nНа основе текста одним словом ответь на вопрос: Местоимение во фразе "{span2}" относится к "{span1}?'
DANETQA_SOURCE_TEMPLATE = 'Контекст: {passage}\nВопрос: {question}\nОтветь "да" или "нет"'
LIDIRUS_SOURCE_TEMPLATE = '''Текст: {sentence1}. Утверждение: {sentence2}
Используя текст, ответь одним словом на вопрос: Вероятно ли утверждение при условии остального текста?'''
RUSSE_SOURCE_TEMPLATE = '''Ответь только "да" или "нет" на вопрос:
В текстовом фрагменте "{sentence1}" и текстовом фрагменте "{sentence2}" означают ли слова "{word}" одно и то же?'''


def get_danetqa(split):
    dataset = load_dataset(HF_DATASET, "danetqa", split=split)
    for row in dataset:
        record = {
            "task": "parus",
            "source": DANETQA_SOURCE_TEMPLATE.format(
                passage=row["passage"],
                question=row["question"]
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = "да" if label == 1 else "нет"
        yield record


def get_muserc(split):
    dataset = load_dataset(HF_DATASET, "muserc", split=split)
    for row in dataset:
        record = {
            "task": "muserc",
            "source": MUSERC_SOURCE_TEMPLATE.format(
                text=row["paragraph"],
                question=row["question"],
                answer=row["answer"]
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = "да" if label == 1 else "нет"
        yield record


def get_parus(split):
    dataset = load_dataset(HF_DATASET, "parus", split=split)
    for row in dataset:
        is_cause = row["question"] == "cause"
        c1 = row["choice1"].rstrip(".").lower()
        c2 = row["choice2"].rstrip(".").lower()
        premise = row["premise"].rstrip(".")

        template = PARUS_CAUSE_SOURCE_TEMPLATE if is_cause else PARUS_EFFECT_SOURCE_TEMPLATE
        record = {
            "task": "parus",
            "source": template.format(
                choice1=c1,
                choice2=c2,
                premise=premise
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = c1 if label == 0 else c2
        yield record


RCB_TARGET_MAPPING = {
    0: "да",
    1: "нет",
    2: "может быть"
}


def get_rcb(split):
    dataset = load_dataset(HF_DATASET, "rcb", split=split)
    for row in dataset:
        record = {
            "task": "rcb",
            "source": RCB_SOURCE_TEMPLATE.format(
                premise=row["premise"],
                question=row["hypothesis"].rstrip(".") + "?"
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = RCB_TARGET_MAPPING[label]
        yield record


def get_rucos(split, sample_rate: float = 0.1):
    dataset = load_dataset(HF_DATASET, "rucos", split=split)
    for row in dataset:
        if split != "test" and random.random() > sample_rate:
            continue
        query = row["query"]
        query = query.replace("@placeholder", RUCOS_MASK)
        text = rucos_clean_text(row["passage"])
        record = {
            "task": "rucos",
            "source": RUCOS_SOURCE_TEMPLATE.format(
                text=text,
                query=query,
                mask=RUCOS_MASK
            )
        }
        if row["answers"]:
            answer = row["answers"][0]
            record["target"] = answer
        yield record


def get_russe(split, sample_rate: float = 0.2):
    dataset = load_dataset(HF_DATASET, "russe", split=split)
    for row in dataset:
        if split != "test" and random.random() > sample_rate:
            continue
        record = {
            "task": "russe",
            "source": RUSSE_SOURCE_TEMPLATE.format(
                sentence1=row["sentence1"],
                sentence2=row["sentence2"],
                word=row["word"]
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = "да" if label == 1 else "нет"
        yield record


def get_rwsd(split):
    dataset = load_dataset(HF_DATASET, "rwsd", split=split)
    for row in dataset:
        record = {
            "task": "rwsd",
            "source": RWSD_SOURCE_TEMPLATE.format(
                text=row["text"],
                span1=row["span1_text"],
                span2=row["span2_text"]
            )
        }
        label = row["label"]
        if label == 1:
            record["target"] = row["span1_text"]
            #record["target"] = "да"
            yield record
        elif label == 0:
            pass
            #record["target"] = "нет"
            #yield record
        elif label == -1:
            yield record


def get_terra(split):
    dataset = load_dataset(HF_DATASET, "terra", split=split)
    for row in dataset:
        record = {
            "task": "terra",
            "source": TERRA_SOURCE_TEMPLATE.format(
                premise=row["premise"],
                hypothesis=row["hypothesis"]
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = "да" if label == 0 else "нет"
        yield record


def get_lidirus():
    dataset = load_dataset(HF_DATASET, "lidirus", split="test")
    for row in dataset:
        record = {
            "task": "terra",
            "source": LIDIRUS_SOURCE_TEMPLATE.format(
                sentence1=row["sentence1"],
                sentence2=row["sentence2"]
            )
        }
        label = row["label"]
        if label != -1:
            record["target"] = "да" if label == 0 else "нет"
        yield record


ALL_TASKS = ("danetqa", "lidirus", "muserc", "parus", "rcb", "rucos", "russe", "rwsd", "terra")

def convert_rsg(split, output_path, tasks: List[str] = ALL_TASKS, use_short: bool = True):
    functions = []
    if "danetqa" in tasks:
        functions.append(get_danetqa(split))
    if "muserc" in tasks:
        functions.append(get_muserc(split))
    if "parus" in tasks:
        functions.append(get_parus(split))
    if "rcb" in tasks:
        functions.append(get_rcb(split))
    if "rucos" in tasks:
        sample_rate = 0.1 if use_short else 1.0
        functions.append(get_rucos(split, sample_rate=sample_rate))
    if "russe" in tasks:
        sample_rate = 0.2 if use_short else 1.0
        functions.append(get_russe(split, sample_rate=sample_rate))
    if "rwsd" in tasks:
        functions.append(get_rwsd(split))
    if "terra" in tasks:
        functions.append(get_terra(split))
    if "lidirus" in tasks and split == "test":
        functions.append(get_lidirus())
    records = [r for r in chain(*functions)]
    for r in records:
        r["source"] = "Задание: {}\n{}".format(r.pop("task"), r.pop("source"))
        r["messages"] = [
            {"role": "user", "content": r.pop("source")},
            {"role": "bot", "content": r.pop("target", None)}
        ]
    random.shuffle(records)
    write_jsonl(records, output_path)


if __name__ == "__main__":
    fire.Fire(convert_rsg)
