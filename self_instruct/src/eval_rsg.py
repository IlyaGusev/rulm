from typing import Tuple
import re
import copy
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import fire
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from nltk import edit_distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

from src.util.io import write_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments

HF_DATASET = "RussianNLP/russian_super_glue"


def generate(
    model,
    tokenizer,
    prompts,
    generation_config,
    debug: bool = True
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        if debug:
            print(tokenizer.decode(sample_input_ids, skip_special_tokens=True))
            print(sample_output)
            print()
        outputs.append(sample_output)
    return outputs


def calc_loss(
    model,
    tokenizer,
    prompts,
    debug: bool = True,
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    labels = data["input_ids"].clone()
    with torch.no_grad():
        outputs = model.forward(**data, return_dict=True)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.transpose(1, 2)

    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)

    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)
    losses = loss.mean(-1)
    if debug:
        for prompt, loss in zip(prompts, losses):
            print(prompt)
            print("Loss:", loss.item())
            print()
    return losses


def predict_saiga_zero_shot(
    model,
    tokenizer,
    generation_config,
    template_path,
    prompts,
    max_prompt_tokens: int = None,
    debug: bool = False
):
    default_conversation = Conversation.from_template(template_path)
    clean_prompts = []
    for prompt in prompts:
        conversation = copy.deepcopy(default_conversation)
        conversation.add_user_message(prompt)
        prompt = conversation.get_prompt(tokenizer, max_tokens=max_prompt_tokens)
        clean_prompts.append(prompt)
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompts=clean_prompts,
        generation_config=generation_config,
        debug=debug
    )


def predict_saiga_zero_shot_logits(
    model,
    tokenizer,
    template_path,
    all_messages,
    max_prompt_tokens: int = None,
    debug: bool = False
):
    default_conversation = Conversation.from_template(template_path)
    clean_prompts = []
    for messages in all_messages:
        conversation = copy.deepcopy(default_conversation)
        for message in messages:
            if message["role"] == "user":
                conversation.add_user_message(message["content"])
            elif message["role"] == "bot":
                conversation.add_bot_message(message["content"])
        prompt = conversation.get_prompt(
            tokenizer,
            max_tokens=max_prompt_tokens,
            add_suffix=False
        )
        clean_prompts.append(prompt)
    return calc_loss(
        model=model,
        tokenizer=tokenizer,
        prompts=clean_prompts,
        debug=debug
    )


def find_lcs(s1, s2):
    max_lcs = ""
    for i in range(len(s1)):
        for j in range(i + 1, len(s1)):
            ss1 = s1[i:j]
            if ss1 in s2 and len(ss1) > len(max_lcs):
                max_lcs = ss1
    return max_lcs

# DaNetQA


DANETQA_PROMPT = '''Контекст: {passage}

Используя контекст, ответь одним словом на вопрос: {question}'''

DANETQA_YES_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|правда|может)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
DANETQA_NO_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*нет",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def clean_danetqa_response(response):
    result = False
    if bool(DANETQA_YES_RE.match(response)):
        result = True
    elif bool(DANETQA_NO_RE.match(response)):
        result = False
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool:", response)
    return result


def predict_danetqa(
    split,
    predict_func,
    output_path,
    batch_size: int = 4,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "danetqa", split=split))
    if nrows:
        records = records[:nrows]

    prompts = []
    for record in records:
        prompt = DANETQA_PROMPT.format(passage=record["passage"], question=record["question"])
        prompts.append(prompt)

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    labels, predictions = [], []
    for record, response in zip(records, responses):
        prediction = clean_danetqa_response(response)
        record["prediction"] = prediction
        label = record["label"]
        if label != -1:
            labels.append(label)
            predictions.append(prediction)

    if labels:
        print("danetqa accuracy:", accuracy_score(labels, predictions))

    outputs = []
    for record in records:
        label = str(record["prediction"]).lower()
        outputs.append({"idx": record["idx"], "label": label})
    write_jsonl(outputs, output_path)

    return records

# TERRA


TERRA_PROMPT = '''Текст: {premise}. Утверждение: {hypothesis}
Используя текст, ответь одним словом на вопрос: Вероятно ли утверждение при условии остального текста?'''

TERRA_ENTAILMENT_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|правда|может|являются)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
TERRA_NOT_ENTAILMENT_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(нет|неверно|неверное|невероятно)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def terra_to_bool(response):
    return response == "entailment"


def clean_terra_response(response):
    result = "not_entailment"
    if bool(TERRA_ENTAILMENT_RE.match(response)):
        result = "entailment"
    elif bool(TERRA_NOT_ENTAILMENT_RE.match(response)):
        result = "not_entailment"
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool", response)
    return result


def predict_terra(
    split,
    predict_func,
    output_path,
    batch_size: int = 8,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "terra", split=split))
    if nrows:
        records = records[:nrows]

    prompts = []
    for record in records:
        prompts.append(TERRA_PROMPT.format(
            premise=record["premise"],
            hypothesis=record["hypothesis"]
        ))

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    labels, predictions = [], []
    for record, response in zip(records, responses):
        prediction = clean_terra_response(response)
        record["prediction"] = prediction
        label = record["label"]
        if label != -1:
            labels.append(1 - label)
            predictions.append(terra_to_bool(prediction))

    if labels:
        print("terra accuracy:", accuracy_score(labels,  predictions))

    outputs = [{"idx": r["idx"], "label": r["prediction"]} for r in records]
    write_jsonl(outputs, output_path)

    return records

# RWSD


RWSD_PROMPT = 'Текст: "{text}"\nНа основе текста одним словом ответь на вопрос: К кому или к чему относится местоимение во фразе "{span2}"?'


def clean_rwsd_response(response, span1):
    lcs = find_lcs(span1.lower(), response.lower())
    return len(lcs) >= 3


def predict_rwsd(
    split,
    predict_func,
    output_path,
    batch_size: int = 4,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "rwsd", split=split))
    if nrows:
        records = records[:nrows]

    prompts = []
    for record in records:
        prompts.append(RWSD_PROMPT.format(
            text=record["text"],
            span2=record["span2_text"],
            span1=record["span1_text"],
        ))

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    labels, predictions = [], []
    for record, response in zip(records, responses):
        prediction = clean_rwsd_response(response, record["span1_text"])
        record["prediction"] = prediction
        label = record["label"]
        if label != -1:
            labels.append(label)
            predictions.append(prediction)

    if labels:
        print("rwsd accuracy:", accuracy_score(labels, predictions))

    outputs = [{"idx": r["idx"], "label": str(r["prediction"])} for r in records]
    write_jsonl(outputs, output_path)

    return records

# MUSERC


MUSERC_PROMPT_STEP1 = """Контекст: {text}

Используя контекст, коротко ответь на вопрос: {question}"""

MUSERC_PROMPT_STEP2 = """Вопрос: {question}
Ответ 1: {answer}
Ответ 2: {predicted_answer}

Ответь одним словом: Полностью совпадают ли Ответ 1 и Ответ 2?
"""


MUSERC_YES_RE = re.compile(
    r"^[^\w]*(да|совпадают)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
MUSERC_NO_RE = re.compile(
    r"^[^\w]*(нет|не совпадают)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def clean_muserc_response(response):
    result = False
    if bool(MUSERC_YES_RE.match(response)):
        result = True
    elif bool(MUSERC_NO_RE.match(response)):
        result = False
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool:", response)
    return result


def predict_muserc(
    split,
    predict_func,
    output_path,
    batch_size: int = 2,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "muserc", split=split))
    if nrows:
        records = records[:nrows]

    prompts = dict()
    prompt2key = dict()
    for record in records:
        text, question = record["paragraph"], record["question"]
        prompt = MUSERC_PROMPT_STEP1.format(text=text, question=question)
        prompts[(text, question)] = prompt
        prompt2key[prompt] = (text, question)
    prompts = list(set(prompts.values()))

    responses = dict()
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_func(batch)
        for response, prompt in zip(raw_responses, batch):
            text, question = prompt2key[prompt]
            responses[(text, question)] = response

    # Step 2
    prompts = []
    for record in records:
        predicted_answer = responses[(record["paragraph"], record["question"])]
        prompt = MUSERC_PROMPT_STEP2.format(
            text=record["paragraph"],
            question=record["question"],
            answer=record["answer"],
            predicted_answer=predicted_answer
        )
        prompts.append(prompt)
    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_func(batch)
        responses.extend([r for r in raw_responses])

    labels, predictions = [], []
    for record, response in zip(records, responses):
        record["prediction"] = clean_muserc_response(response)
        if record["label"] != -1:
            labels.append(record["label"])
            predictions.append(record["prediction"])

    if labels:
        print("muserc accuracy:", accuracy_score(labels, predictions))

    outputs = []
    prev_idx = None
    for record in records:
        idx = record["idx"]
        pidx, qidx, aidx = idx["paragraph"], idx["question"], idx["answer"]
        ppidx, pqidx = None, None
        if prev_idx:
            ppidx, pqidx = prev_idx["paragraph"], prev_idx["question"]

        if ppidx != pidx:
            outputs.append({"idx": pidx, "passage": {"questions": []}})
        paragraph = outputs[-1]

        if pqidx != qidx:
            paragraph["passage"]["questions"].append({"idx": qidx, "answers": []})
        question = paragraph["passage"]["questions"][-1]

        answer = {"idx": aidx, "label": int(record["prediction"])}
        question["answers"].append(answer)
        prev_idx = idx

    write_jsonl(outputs, output_path)
    return records

# RUCOS


def clean_rucos_response(response, entities):
    entities.sort(key=lambda x: edit_distance(response, x))
    return entities[0]


def rucos_clean_text(text):
    text = " ".join([s.strip() for s in text.split("@header")]).strip()
    return [s.strip() for s in text.split("@highlight")][0].strip()


def predict_rucos(
    split,
    predict_func,
    output_path,
    batch_size: int = 4,
    nrows: int = None,
    debug: bool = False
):
    records = list(load_dataset(HF_DATASET, "rucos", split=split))
    if nrows:
        records = records[:nrows]

    prompts = list()
    for record in records:
        text = record["passage"]
        entities = record["entities"]
        query = record["query"]
        text = rucos_clean_text(text)
        for answer in entities:
            query_answer = query.replace("@placeholder", answer.strip())
            prompts.append({
                "text": text,
                "query": query,
                "message": text + " " + query_answer,
                "answer": answer
            })

    responses = defaultdict(list)
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        batch_messages = [[
            {"role": "user", "content": p["message"]},
        ] for p in batch]
        losses = predict_func(batch_messages)
        for loss, data in zip(losses, batch):
            responses[(data["text"], data["query"])].append((loss, data["answer"]))

    final_responses = dict()
    for (text, query), answers in responses.items():
        if debug:
            print("Text:", text)
            print("Query:", query)
            for loss, answer in sorted(answers):
                print(loss.item(), answer)
            print()
        final_responses[(text, query)] = min(answers)[1]

    all_count, correct_count = 0, 0
    for record in records:
        text = record["passage"]
        entities = record["entities"]
        query = record["query"]
        text = rucos_clean_text(text)
        response = final_responses[(text, query)]
        record["prediction"] = clean_rucos_response(response, entities)
        answers = record["answers"]
        if answers:
            all_count += 1
            for answer in answers:
                if answer.strip().lower() == record["prediction"].strip().lower():
                    correct_count += 1
                    break
    if all_count > 0:
        print("rucos accuracy:", correct_count / all_count)

    outputs = [{"idx": r["idx"]["query"], "label": r["prediction"]} for r in records]
    write_jsonl(outputs, output_path)

    return records

# LIDIRUS


LIDIRUS_PROMPT = '''Текст: "{sentence1}"

Используя текст, можно ли сказать, что утверждение "{sentence2}" точно корректно относительно ситуации из текста? Ответь только "да" или "нет".'''

LIDIRUS_ENTAILMENT_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|правда|может|вероятна|верная)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
LIDIRUS_NOT_ENTAILMENT_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(не|нет|неверно|неверное|невероятна|неверная)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def lidirus_to_bool(response):
    return response == "entailment"


def clean_lidirus_response(response):
    result = "not_entailment"
    if bool(LIDIRUS_ENTAILMENT_RE.match(response)):
        result = "entailment"
    elif bool(LIDIRUS_NOT_ENTAILMENT_RE.match(response)):
        result = "not_entailment"
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool", response)
    return result


def predict_lidirus(
    predict_func,
    output_path,
    batch_size: int = 4,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "lidirus", split="test"))
    if nrows:
        records = records[:nrows]

    prompts = [LIDIRUS_PROMPT.format(
        sentence1=r["sentence1"],
        sentence2=r["sentence2"]
    ) for r in records]

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    labels, predictions = [], []
    for record, response in zip(records, responses):
        prediction = clean_lidirus_response(response)
        record["prediction"] = prediction
        label = record["label"]
        labels.append(lidirus_to_bool(label))
        predictions.append(lidirus_to_bool(prediction))

    print("lidirus accuracy:", accuracy_score(labels, predictions))
    print("lidirus corr:", matthews_corrcoef(labels, predictions))

    outputs = [{"idx": r["idx"], "label": r["prediction"]} for r in records]
    write_jsonl(outputs, output_path)

    return records

# PARUS


PARUS_CAUSE_PROMPT = """Выбери одну наиболее вероятную причину исключительно из двух предложенных вариантов.
Делай выбор на основе здравого смысла и своих знаний о мире. Обязательно учитывай саму ситуацию.

Варианты: так как {choice1}; так как {choice2}

{premise}, так как..."""

PARUS_EFFECT_PROMPT = """Выбери одно наиболее вероятное следствие исключительно из двух предложенных вариантов.
Делай выбор на основе здравого смысла и своих знаний о мире. Обязательно учитывай саму ситуацию.

Варианты: поэтому {choice1}; поэтому {choice2}

{premise}, поэтому..."""


def predict_parus(
    split,
    predict_func,
    output_path,
    batch_size: int = 16,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "parus", split=split))
    if nrows:
        records = records[:nrows]

    prompts = []
    for r in records:
        idx = r["idx"]
        c1 = r["choice1"].rstrip(".").lower()
        c2 = r["choice2"].rstrip(".").lower()
        premise = r["premise"].rstrip(".")

        is_cause = r["question"] == "cause"
        template = PARUS_CAUSE_PROMPT if is_cause else PARUS_EFFECT_PROMPT
        prompts.append(template.format(
            premise=premise,
            choice1=c1,
            choice2=c2
        ))

    responses = list()
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    assert len(responses) == len(records)
    for idx, (response, record) in enumerate(zip(responses, records)):
        response = response.lower()
        c1 = record["choice1"].rstrip(".").lower()
        c2 = record["choice2"].rstrip(".").lower()
        c1_lcs = find_lcs(response, c1)
        c2_lcs = find_lcs(response, c2)
        record["prediction"] = int(len(c2_lcs) > len(c1_lcs))

    if records[0]["label"] != -1:
        y_true, y_pred = [], []
        for r in records:
            y_pred.append(r["prediction"])
            y_true.append(r["label"])
        score = accuracy_score(y_true, y_pred)
        print("parus accuracy:", score)

    outputs = [{"idx": r["idx"], "label": int(r["prediction"])} for r in records]
    write_jsonl(outputs, output_path)

    return records

# RCB


RCB_ANSWER_PROMPT = """Дан текст: "{premise}"

Ответь на вопрос по тексту "да", "нет" или "может быть": {question}"""

RCB_YES_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|вероятно)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

RCB_NO_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(нет|неверно|неверное|невероятно|не)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def clean_rcb_response(response):
    is_contradiction = bool(RCB_NO_RE.match(response))
    is_entailment = bool(RCB_YES_RE.match(response))
    if is_contradiction:
        return "contradiction"
    if is_entailment:
        return "entailment"
    return "neutral"


def rcb_label2index(label):
    mapping = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2
    }
    return mapping[label]


def predict_rcb(
    split,
    predict_func,
    output_path,
    batch_size: int = 8,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "rcb", split=split))
    if nrows:
        records = records[:nrows]

    questions = [record["hypothesis"].rstrip(".") + "?" for record in records]

    prompts = []
    for record, question in zip(records, questions):
        prompts.append(RCB_ANSWER_PROMPT.format(
            premise=record["premise"],
            question=question
        ))

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    for r, response in zip(records, responses):
        r["prediction"] = clean_rcb_response(response)

    if records[0]["label"] != -1:
        labels = [r["label"] for r in records]
        responses = [rcb_label2index(r["prediction"]) for r in records]
        print("rcb accuracy:", accuracy_score(labels, responses))

    outputs = [{"idx": r["idx"], "label": r["prediction"]} for r in records]
    write_jsonl(outputs, output_path)

    return records


# RUSSE
RUSSE_PROMPT = 'Слово "{word}" в предложении "{sentence1}" и слово "{word}" в предложении "{sentence2}" означают одно и то же?'

RUSSE_YES_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|вероятно)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def clean_russe_response(response):
    if bool(RUSSE_YES_RE.match(response)):
        return 1
    return 0


def predict_russe(
    split,
    predict_func,
    output_path,
    batch_size: int = 8,
    nrows: int = None
):
    records = list(load_dataset(HF_DATASET, "russe", split=split))
    if nrows:
        records = records[:nrows]

    prompts = []
    for record in records:
        prompts.append(RUSSE_PROMPT.format(
            sentence1=record["sentence1"],
            sentence2=record["sentence2"],
            word=record["word"]
        ))

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    for r, response in zip(records, responses):
        r["prediction"] = clean_russe_response(response)

    if records[0]["label"] != -1:
        labels = [r["label"] for r in records]
        responses = [r["prediction"] for r in records]
        print("russe accuracy:", accuracy_score(labels, responses))

    outputs = [{
        "idx": r["idx"],
        "label": str(bool(r["prediction"])).lower()
    } for r in records]
    write_jsonl(outputs, output_path)

    return records


ALL_TASKS = ("danetqa", "lidirus", "muserc", "parus", "rcb", "rucos", "russe", "rwsd", "terra")


def main(
    model_name,
    nrows: int = None,
    template_path: str = "internal_prompts/saiga_v2.json",
    split: str = "test",
    predictions_dir: str = "submission",
    debug: bool = False,
    tasks: Tuple[str] = ALL_TASKS
):
    predictions_dir = Path(predictions_dir)

    predict_short = None
    predict_long = None
    predict_logits = None

    if model_name not in ("gpt-4", "gpt-3.5-turbo"):
        model, tokenizer, generation_config = load_saiga(model_name)
        generation_config.no_repeat_ngram_size = 64
        generation_config.temperature = 0.01

        def predict_saiga_zero_shot_bound(batch):
            generation_config.max_new_tokens = 256
            return predict_saiga_zero_shot(
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                template_path=template_path,
                prompts=batch,
                debug=debug
            )

        def predict_saiga_zero_shot_bound_short(batch):
            generation_config.max_new_tokens = 8
            return predict_saiga_zero_shot(
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                template_path=template_path,
                prompts=batch,
                debug=debug
            )

        def predict_saiga_zero_shot_logits_bound(batch_messages):
            return predict_saiga_zero_shot_logits(
                model=model,
                tokenizer=tokenizer,
                template_path=template_path,
                all_messages=batch_messages,
                debug=debug
            )

        predict_long = predict_saiga_zero_shot_bound
        predict_short = predict_saiga_zero_shot_bound_short
        predict_logits = predict_saiga_zero_shot_logits_bound

    else:
        def predict_chatgpt(batch):
            batch = [[{"role": "user", "content": prompt}] for prompt in batch]
            responses = openai_batch_completion(batch, model_name=model_name)
            responses = [r.message.content for r in responses]
            return responses

        def predict_chatgpt_short(batch):
            batch = [[{"role": "user", "content": prompt}] for prompt in batch]
            responses = openai_batch_completion(
                batch,
                decoding_args=OpenAIDecodingArguments(max_tokens=16),
                model_name=model_name
            )
            responses = [r.message.content for r in responses]
            return responses

        predict_long = predict_chatgpt
        predict_short = predict_chatgpt_short

    if "danetqa" in tasks:
        predict_danetqa(
            split=split,
            predict_func=predict_short,
            output_path=predictions_dir / "DaNetQA.jsonl",
            nrows=nrows
        )

    if "terra" in tasks:
        predict_terra(
            split=split,
            predict_func=predict_short,
            output_path=predictions_dir / "TERRa.jsonl",
            nrows=nrows
        )

    if "rwsd" in tasks:
        predict_rwsd(
            split=split,
            predict_func=predict_long,
            output_path=predictions_dir / "RWSD.jsonl",
            nrows=nrows
        )

    if "rucos" in tasks:
        assert predict_logits is not None
        predict_rucos(
            split=split,
            predict_func=predict_logits,
            output_path=predictions_dir / "RuCoS.jsonl",
            nrows=nrows
        )

    if "lidirus" in tasks:
        predict_lidirus(
            predict_func=predict_short,
            output_path=predictions_dir / "LiDiRus.jsonl",
            nrows=nrows
        )
    if "parus" in tasks:
        predict_parus(
            split=split,
            predict_func=predict_long,
            output_path=predictions_dir / "PARus.jsonl",
            nrows=nrows
        )
    if "rcb" in tasks:
        predict_rcb(
            split=split,
            predict_func=predict_long,
            output_path=predictions_dir / "RCB.jsonl",
            nrows=nrows
        )
    if "russe" in tasks:
        predict_russe(
            split=split,
            predict_func=predict_short,
            output_path=predictions_dir / "RUSSE.jsonl",
            nrows=nrows
        )
    if "muserc" in tasks:
        predict_muserc(
            split=split,
            predict_func=predict_long,
            output_path=predictions_dir / "MuSeRC.jsonl",
            nrows=nrows
        )


if __name__ == "__main__":
    fire.Fire(main)
