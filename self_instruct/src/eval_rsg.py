import json
import os
import re
import copy
from tqdm import tqdm
from collections import defaultdict
from difflib import SequenceMatcher

import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from nltk import edit_distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

from src.util.io import read_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga

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
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 64,
    temperature: float = 0.01
):
    generation_config.max_new_tokens = max_new_tokens
    generation_config.no_repeat_ngram_size = no_repeat_ngram_size
    generation_config.temperature = temperature

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
        generation_config=generation_config
    )


def predict_saiga_zero_shot_logits(
    model,
    tokenizer,
    template_path,
    all_messages,
    max_prompt_tokens: int = None
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
        prompts=clean_prompts
    )


def find_lcs(s1, s2):
    max_lcs = ""
    for i in range(len(s1)):
        for j in range(i + 1, len(s1)):
            ss1 = s1[i:j]
            if ss1 in s2 and len(ss1) > len(max_lcs):
                max_lcs = ss1
    return max_lcs


DANETQA_PROMPT = '''Контекст: {passage}

Используя контекст, ответь одним словом на вопрос: {question}'''

DANETQA_YES_RE = re.compile(
    "^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|правда|может)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
DANETQA_NO_RE = re.compile(
    "^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*нет",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

def clean_danetqa_response(response):
    result = None
    if bool(DANETQA_YES_RE.match(response)):
        result = True
    elif bool(DANETQA_NO_RE.match(response)):
        result = False
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool:", response)
        result = False
    return result


def predict_danetqa(
    test_path,
    predict_func,
    batch_size: int = 4
):
    prompts = []
    records = list(read_jsonl(test_path))
    for record in records:
        prompt = DANETQA_PROMPT.format(passage=record["passage"], question=record["question"])
        prompts.append(prompt)

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_func(batch)
        responses.extend([clean_danetqa_response(r) for r in raw_responses])
    for record, response in zip(records, responses):
        record["prediction"] = response
    if "label" in records[0]:
        labels = [r["label"] for r in records]
        responses = [r["prediction"] for r in records]
        print("danetqa accuracy:", accuracy_score(labels, responses))
    return records


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
    result = None
    if bool(TERRA_ENTAILMENT_RE.match(response)):
        result = "entailment"
    elif bool(TERRA_NOT_ENTAILMENT_RE.match(response)):
        result = "not_entailment"
    else:
        result = "not_entailment"
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool", response)
    return result


def predict_terra(
    test_path,
    predict_func,
    batch_size: int = 4
):
    prompts = []
    records = list(read_jsonl(test_path))
    for record in records:
        prompt = TERRA_PROMPT.format(
            premise=record["premise"],
            hypothesis=record["hypothesis"]
        )
        prompts.append(prompt)
    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))
    for record, response in zip(records, responses):
        record["prediction"] = clean_terra_response(response)
    if "label" in records[0]:
        labels = [terra_to_bool(r["label"]) for r in records]
        responses = [terra_to_bool(r["prediction"]) for r in records]
        print("terra accuracy:", accuracy_score(labels, responses))
    return records

RWSD_PROMPT = 'Текст: "{text}"\nНа основе текста одним словом ответь на вопрос: К кому или к чему относится местоимение во фразе "{span2}"?'

RWSD_YES_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
RWSD_NO_RE = re.compile(
    r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*нет",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def clean_rwsd_response(response, span1):
    span1 = span1.lower()
    response = response.lower()
    size = 0
    for i in range(len(span1)):
        for j in range(i + 1, len(span1)):
            for k in range(len(response)):
                for l in range(k + 1, len(response)):
                    ss1 = span1[i:j]
                    ss2 = response[k:l]
                    if ss1 == ss2:
                        size = max(size, len(ss1))
    print(span1, "#", response, "#", size)
    return size >= 3


def predict_rwsd(
    test_path,
    predict_func,
    batch_size: int = 4
):
    prompts = []
    records = list(read_jsonl(test_path))
    for record in records:
        prompt = RWSD_PROMPT.format(
            text=record["text"],
            span2=record["target"]["span2_text"],
            span1=record["target"]["span1_text"],
        )
        prompts.append(prompt)
    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))
    for record, response in zip(records, responses):
        record["prediction"] = clean_rwsd_response(response, record["target"]["span1_text"])
    if "label" in records[0]:
        labels = [r["label"] for r in records]
        responses = [r["prediction"] for r in records]
        print("rwsd accuracy:", accuracy_score(labels, responses))
    return records

MUSERC_PROMPT_STEP1 = """Контекст: {text}

Используя контекст, коротко ответь на вопрос: {question}"""

MUSERC_PROMPT_STEP2 = """Вопрос: {question}
Ответ 1: {answer}
Ответ 2: {predicted_answer}

Ответь одним словом: Полностью совпадают ли Ответ 1 и Ответ 2?
"""


MUSERC_YES_RE = re.compile(
    "^[^\w]*(да|совпадают)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
MUSERC_NO_RE = re.compile(
    "^[^\w]*(нет|не совпадают)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

def clean_muserc_response(response):
    result = None
    if bool(MUSERC_YES_RE.match(response)):
        result = True
    elif bool(MUSERC_NO_RE.match(response)):
        result = False
    else:
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool:", response)
        result = False
    return result


def predict_muserc(
    test_path,
    predict_func,
    batch_size: int = 2
):
    records = list(read_jsonl(test_path))
    clean_records = []
    for record in records:
        record = record["passage"]
        text = record["text"]
        for question_record in record["questions"]:
            question = question_record["question"]
            for answer_record in question_record["answers"]:
                answer = answer_record["text"]
                clean_records.append({
                    "text": text,
                    "question": question,
                    "answer": answer
                })
    prompts = dict()
    prompt2key = dict()
    for record in clean_records:
        text, question = record["text"], record["question"]
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
    for record in clean_records:
        predicted_answer = responses[(record["text"], record["question"])]
        prompt = MUSERC_PROMPT_STEP2.format(
            text=record["text"],
            question=record["question"],
            answer=record["answer"],
            predicted_answer=predicted_answer
        )
        prompts.append(prompt)
    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_func(batch)
        responses.extend([r for r in raw_responses])

    index = 0
    predictions = []
    labels = []
    for record in records:
        record = record["passage"]
        for question_record in record["questions"]:
            for answer_record in question_record["answers"]:
                label = answer_record.get("label")
                response = clean_muserc_response(responses[index])
                answer_record["prediction"] = response
                predictions.append(response)
                if label is not None:
                    labels.append(label)
                index += 1

    if labels:
        print("muserc accuracy:", accuracy_score(labels, predictions))
    return records


def clean_rucos_response(response, text, entities):
    for e in entities:
        e["text"] = text[e["start"]:e["end"]].strip()
        e["distance"] = edit_distance(response, e["text"])
    entities.sort(key=lambda x: x["distance"])
    return entities[0]["text"]


def rucos_clean_text(text):
    text = " ".join([s.strip() for s in text.split("@header")]).strip()
    return [s.strip() for s in text.split("@highlight")][0].strip()


def predict_rucos(
    test_path,
    predict_func,
    batch_size: int = 4
):
    records = list(read_jsonl(test_path))

    prompts = list()
    for record in records:
        text = record["passage"]["text"]
        entities = record["passage"]["entities"]
        entities = [text[e["start"]:e["end"]].strip() for e in entities]
        text = rucos_clean_text(text)
        for qas in record["qas"]:
            query = qas["query"]
            for answer in entities:
                query_answer = query.replace("@placeholder", answer)
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
        print("Text:", text)
        print("Query:", query)
        for loss, answer in sorted(answers):
            print(loss.item(), answer)
        print()
        final_responses[(text, query)] = min(answers)[1]

    all_count = 0
    correct_count = 0
    for record in records:
        text = rucos_clean_text(record["passage"]["text"])
        entities = record["passage"]["entities"]
        for qas in record["qas"]:
            query = qas["query"]
            response = final_responses[(text, query)]
            qas["prediction"] = clean_rucos_response(response, text, entities)
            answers = qas.get("answers")
            if answers:
                all_count += 1
                for answer in answers:
                    if answer["text"] == qas["prediction"]:
                        correct_count += 1
                        break
    if all_count > 0:
        print("rucos accuracy:", correct_count / all_count)
    return records

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
    result = None
    if bool(LIDIRUS_ENTAILMENT_RE.match(response)):
        result = "entailment"
    elif bool(LIDIRUS_NOT_ENTAILMENT_RE.match(response)):
        result = "not_entailment"
    else:
        result = "not_entailment"
        print("ERROR! Не удалось найти Да/Нет в ответе модели и преобразовать его в bool", response)
    return result


def predict_lidirus(
    test_path,
    predict_func,
    batch_size: int = 4
):
    records = list(read_jsonl(test_path))
    prompts = [LIDIRUS_PROMPT.format(
        sentence1=r["sentence1"],
        sentence2=r["sentence2"]
    ) for r in records]

    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        responses.extend(predict_func(batch))

    for record, response in zip(records, responses):
        record["prediction"] = clean_lidirus_response(response)

    if "label" in records[0]:
        labels = [lidirus_to_bool(r["label"]) for r in records]
        responses = [lidirus_to_bool(r["prediction"]) for r in records]
        print("lidirus accuracy:", accuracy_score(labels, responses))
        print("lidirus corr:", matthews_corrcoef(labels, responses))
    return records

#PARUS_CAUSE_PROMPT = """{premise} По какой причине это случилось? Варианты: "(A) {choice1}" или "(B) {choice2}"."""
#PARUS_EFFECT_PROMPT = """{premise} К каким последствиям это привело? Варианты: "(A) {choice1}" или "(B) {choice2}"."""

#PARUS_CAUSE_PROMPT = """Выбери одну наиболее вероятную причину исключительно из двух предложенных вариантов.
#
#Варианты: так как {choice1}; так как {choice2}
#
#{premise}, так как..."""

#PARUS_EFFECT_PROMPT = """Выбери одно наиболее вероятное следствие исключительно из двух предложенных вариантов.
#
#Варианты: поэтому {choice1}; поэтому {choice2}
#
#{premise}, поэтому..."""

PARUS_CAUSE_PROMPT = """Выбери одну наиболее вероятную причину исключительно из двух предложенных вариантов.
Делай выбор на основе здравого смысла и своих знаний о мире. Обязательно учитывай саму ситуацию.

Варианты: так как {choice1}; так как {choice2}

{premise}, так как..."""

PARUS_EFFECT_PROMPT = """Выбери одно наиболее вероятное следствие исключительно из двух предложенных вариантов.
Делай выбор на основе здравого смысла и своих знаний о мире. Обязательно учитывай саму ситуацию.

Варианты: поэтому {choice1}; поэтому {choice2}

{premise}, поэтому..."""


def predict_parus(split, predict_func, batch_size: int = 8):
    dataset = list(load_dataset(HF_DATASET, "parus", split=split))

    prompts = []
    for r in dataset:
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

    assert len(responses) == len(dataset)
    for idx, (response, record) in enumerate(zip(responses, dataset)):
        response = response.lower()
        c1 = record["choice1"].rstrip(".").lower()
        c2 = record["choice2"].rstrip(".").lower()
        c1_lcs = find_lcs(response, c1)
        c2_lcs = find_lcs(response, c2)
        record["prediction"] = int(len(c2_lcs) > len(c1_lcs))
        if record["prediction"] != record["label"]:
            true_response = c1
            if record["label"] == 1:
                true_response = c2
            print(record["premise"], "##", true_response, "##", response)

    if "label" in dataset[0]:
        y_true, y_pred = [], []
        for r in dataset:
            y_pred.append(r["prediction"])
            y_true.append(r.get("label"))
        score = accuracy_score(y_true, y_pred)
        print("parus accuracy:", score)
        with open("parus_score.txt", "w") as w:
            w.write(str(score) + "\n")
    return dataset


def main(
    model_name,
    template_path,
    data_dir,
    split,
    predictions_dir
):
    model, tokenizer, generation_config = load_saiga(model_name)

    def predict_saiga_zero_shot_bound(batch):
        return predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )

    def predict_saiga_zero_shot_logits_bound(batch_messages):
        return predict_saiga_zero_shot_logits(
            model=model,
            tokenizer=tokenizer,
            template_path=template_path,
            all_messages=batch_messages
        )

    #danetqa_test_path = os.path.join(data_dir, "DaNetQA", split + ".jsonl")
    #danetqa_predictions = predict_danetqa(
    #    danetqa_test_path,
    #    predict_func=predict_saiga_zero_shot_bound
    #)
    #danetqa_result_path = os.path.join(predictions_dir, "DaNetQA.jsonl")
    #with open(danetqa_result_path, "w") as w:
    #    for record in danetqa_predictions:
    #        label = str(record["prediction"]).lower()
    #        w.write(json.dumps({"idx": record["idx"], "label": label}) + "\n")

    #terra_test_path = os.path.join(data_dir, "TERRa", split + ".jsonl")
    #terra_predictions = predict_terra(
    #    terra_test_path,
    #    predict_func=predict_saiga_zero_shot_bound
    #)
    #terra_result_path = os.path.join(predictions_dir, "TERRa.jsonl")
    #with open(terra_result_path, "w") as w:
    #    for record in terra_predictions:
    #        w.write(json.dumps({"idx": record["idx"], "label": record["prediction"]}) + "\n")

    #rwsd_test_path = os.path.join(data_dir, "RWSD", split + ".jsonl")
    #rwsd_predictions = predict_rwsd(
    #    rwsd_test_path,
    #    predict_func=predict_saiga_zero_shot_bound
    #)
    #rwsd_result_path = os.path.join(predictions_dir, "RWSD.jsonl")
    #with open(rwsd_result_path, "w") as w:
    #    for record in rwsd_predictions:
    #        label = str(record["prediction"])
    #        w.write(json.dumps({"idx": record["idx"], "label": label}) + "\n")

    #rucos_test_path = os.path.join(data_dir, "RuCoS", split + ".jsonl")
    #rucos_predictions = predict_rucos(
    #    rucos_test_path,
    #    predict_func=predict_saiga_zero_shot_logits_bound
    #)
    #rucos_result_path = os.path.join(predictions_dir, "RuCoS.jsonl")
    #with open(rucos_result_path, "w") as w:
    #    for record in rucos_predictions:
    #        label = record["qas"][0]["prediction"]
    #        idx = record["qas"][0]["idx"]
    #        w.write(json.dumps({"idx": idx, "label": label}, ensure_ascii=False) + "\n")

    #lidirus_test_path = os.path.join(data_dir, "LiDiRus", "LiDiRus.jsonl")
    #lidirus_predictions = predict_lidirus(
    #    lidirus_test_path,
    #    predict_func=predict_saiga_zero_shot_bound
    #)
    #lidirus_result_path = os.path.join(predictions_dir, "LiDiRus.jsonl")
    #with open(lidirus_result_path, "w") as w:
    #    for record in lidirus_predictions:
    #        w.write(json.dumps({"idx": record["idx"], "label": record["prediction"]}) + "\n")

    parus_predictions = predict_parus(
        split=split,
        predict_func=predict_saiga_zero_shot_bound
    )
    parus_result_path = os.path.join(predictions_dir, "PARus.jsonl")
    with open(parus_result_path, "w") as w:
        for r in parus_predictions:
            w.write(json.dumps({"idx": r["idx"], "label": int(r["prediction"])}) + "\n")

    muserc_test_path = os.path.join(data_dir, "MuSeRC", "val" + ".jsonl")
    muserc_predictions = predict_muserc(
        muserc_test_path,
        predict_func=predict_saiga_zero_shot_bound
    )
    muserc_result_path = os.path.join(predictions_dir, "MuSeRC.jsonl")
    with open(muserc_result_path, "w") as w:
        for record in muserc_predictions:
            final_record = {"idx": record["idx"], "passage": {"questions": []}}
            for question in record["passage"]["questions"]:
                clean_question = {"idx": question["idx"], "answers": []}
                for answer in question["answers"]:
                    clean_answer = {"idx": answer["idx"], "label": int(answer["prediction"])}
                    clean_question["answers"].append(clean_answer)
                final_record["passage"]["questions"].append(clean_question)
            w.write(json.dumps(final_record, ensure_ascii=False) + "\n")


main(
    "models/saiga_13b_lora",
    "internal_prompts/saiga_v2.json",
    "data/rsg",
    "validation",
    "submission"
)
