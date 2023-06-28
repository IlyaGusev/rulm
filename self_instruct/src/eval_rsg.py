import json
import os
import re
import copy
from tqdm import tqdm
from difflib import SequenceMatcher

import torch
from nltk import edit_distance
from sklearn.metrics import accuracy_score

from src.util.io import read_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga


def generate(
    model,
    tokenizer,
    prompts,
    generation_config,
    debug: bool = True,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 64
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    generation_config.max_new_tokens = max_new_tokens
    generation_config.no_repeat_ngram_size = no_repeat_ngram_size
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
    debug: bool = True
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    target_ids = data["input_ids"].clone()
    with torch.no_grad():
        outputs = model.forward(**data, return_dict=True, labels=target_ids)
        print(outputs.loss)
    return outputs.loss


def predict_saiga_zero_shot(
    model,
    tokenizer,
    generation_config,
    template_path,
    prompts,
    max_prompt_tokens: int = None
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
        generation_config=generation_config
    )


def predict_saiga_zero_shot_logits(
    model,
    tokenizer,
    template_path,
    prompts,
    answers,
    max_prompt_tokens: int = None
):
    default_conversation = Conversation.from_template(template_path)
    clean_prompts = []
    for prompt, prompt_answers in zip(prompts, answers):
        conversation = copy.deepcopy(default_conversation)
        conversation.add_user_message(prompt)
        for answer in prompt_answers:
            conversation.add_bot_message(answer)
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
    model,
    tokenizer,
    generation_config,
    template_path,
    batch_size: int = 4
):
    prompts = []
    records = list(read_jsonl(test_path))
    for record in records:
        prompt = DANETQA_PROMPT.format(passage=record["passage"], question=record["question"])
        prompts.append(prompt)
    responses = []
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )
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
    model,
    tokenizer,
    generation_config,
    template_path,
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
        raw_responses = predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )
        responses.extend([clean_terra_response(r) for r in raw_responses])
    for record, response in zip(records, responses):
        record["prediction"] = response
    if "label" in records[0]:
        labels = [terra_to_bool(r["label"]) for r in records]
        responses = [terra_to_bool(r["prediction"]) for r in records]
        print("terra accuracy:", accuracy_score(labels, responses))
    return records

#RWSD_PROMPT = 'Текст: "{text}"\nНа основе текста одним словом ответь на вопрос: про кого или про что говорится в фразе "{span2}"?'
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
    model,
    tokenizer,
    generation_config,
    template_path,
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
        raw_responses = predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )
        responses.extend([r for r in raw_responses])
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
    model,
    tokenizer,
    generation_config,
    template_path,
    batch_size: int = 4
):
    records = list(read_jsonl(test_path))[:5]
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
    for record in clean_records:
        prompt = MUSERC_PROMPT_STEP1.format(
            text=record["text"],
            question=record["question"]
        )
        prompts[(record["text"], record["question"])] = prompt

    prompt2key = dict()
    for (text, question), prompt in prompts.items():
        prompt2key[prompt] = (text, question)
    prompts = list(prompts.values())

    responses = dict()
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )
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
        raw_responses = predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch
        )
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
                record["prediction"] = response
                predictions.append(response)
                if label is not None:
                    labels.append(label)
                index += 1

    if labels:
        print("muserc accuracy:", accuracy_score(labels, predictions))
    return records


RUCOS_PROMPT = """Контекст: {text}

Максимально коротко ответь на вопрос: Какая сущность должна быть вместо "@placeholder" в этой фразе: "{query}", согласно контексту?"""


def clean_rucos_response(response, text, entities):
    for e in entities:
        e["text"] = text[e["start"]:e["end"]]
        e["distance"] = edit_distance(response, e["text"])
        print(response, " # ", e["text"], " # ", e["distance"])
    entities.sort(key=lambda x: x["distance"])
    return entities[0]["text"]


def rucos_clean_text(text):
    return text.split("@highlight")[0].strip()


def predict_rucos(
    test_path,
    model,
    tokenizer,
    generation_config,
    template_path,
    batch_size: int = 4
):
    records = list(read_jsonl(test_path))[:10]

    prompts, prompt2key = dict(), dict()
    for record in records:
        text = rucos_clean_text(record["passage"]["text"])
        for qas in record["qas"]:
            query = qas["query"]
            prompt = RUCOS_PROMPT.format(
                text=text,
                query=query
            )
            prompts[(text, query)] = prompt
            prompt2key[prompt] = (text, query)
    prompts = list(prompts.values())

    responses = dict()
    for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
        raw_responses = predict_saiga_zero_shot_logits(
            model=model,
            tokenizer=tokenizer,
            template_path=template_path,
            prompts=batch,
            answers=[["да", "нет"]]
        )
        for response, prompt in zip(raw_responses, batch):
            text, query = prompt2key[prompt]
            responses[(text, query)] = response

    all_count = 0
    correct_count = 0
    for record in records:
        text = rucos_clean_text(record["passage"]["text"])
        entities = record["passage"]["entities"]
        for qas in record["qas"]:
            query = qas["query"]
            response = responses[(text, query)]
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


def main(
    model_name,
    template_path,
    data_dir,
    split,
    predictions_dir
):
    model, tokenizer, generation_config = load_saiga(model_name)

    #danetqa_test_path = os.path.join(data_dir, "DaNetQA", split + ".jsonl")
    #danetqa_predictions = predict_danetqa(
    #    danetqa_test_path,
    #    model=model,
    #    tokenizer=tokenizer,
    #    generation_config=generation_config,
    #    template_path=template_path
    #)
    #danetqa_result_path = os.path.join(predictions_dir, "DaNetQA.jsonl")
    #with open(danetqa_result_path, "w") as w:
    #    for record in danetqa_predictions:
    #        label = str(record["prediction"]).lower()
    #        w.write(json.dumps({"idx": record["idx"], "label": label}) + "\n")

    #terra_test_path = os.path.join(data_dir, "TERRa", split + ".jsonl")
    #terra_predictions = predict_terra(
    #    terra_test_path,
    #    model=model,
    #    tokenizer=tokenizer,
    #    generation_config=generation_config,
    #    template_path=template_path
    #)
    #terra_result_path = os.path.join(predictions_dir, "TERRa.jsonl")
    #with open(terra_result_path, "w") as w:
    #    for record in terra_predictions:
    #        w.write(json.dumps({"idx": record["idx"], "label": record["prediction"]}) + "\n")

    #rwsd_test_path = os.path.join(data_dir, "RWSD", split + ".jsonl")
    #rwsd_predictions = predict_rwsd(
    #    rwsd_test_path,
    #    model=model,
    #    tokenizer=tokenizer,
    #    generation_config=generation_config,
    #    template_path=template_path
    #)
    #rwsd_result_path = os.path.join(predictions_dir, "RWSD.jsonl")
    #with open(rwsd_result_path, "w") as w:
    #    for record in rwsd_predictions:
    #        label = str(record["prediction"])
    #        w.write(json.dumps({"idx": record["idx"], "label": label}) + "\n")

    rucos_test_path = os.path.join(data_dir, "RuCoS", split + ".jsonl")
    rucos_predictions = predict_rucos(
        rucos_test_path,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        template_path=template_path,
    )
    rucos_result_path = os.path.join(predictions_dir, "RuCoS.jsonl")
    with open(rucos_result_path, "w") as w:
        for record in rucos_predictions:
            label = record["qas"][0]["prediction"]
            idx = record["qas"][0]["idx"]
            w.write(json.dumps({"idx": idx, "label": label}) + "\n")

    muserc_test_path = os.path.join(data_dir, "MuSeRC", split + ".jsonl")
    muserc_predictions = predict_muserc(
        muserc_test_path,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        template_path=template_path,
        batch_size=2
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
    "val",
    "submission"
)
