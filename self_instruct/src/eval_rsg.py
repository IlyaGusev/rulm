import re
import copy
from tqdm import tqdm

from sklearn.metrics import accuracy_score

from src.util.io import read_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga


def generate(model, tokenizer, prompts, generation_config, debug: bool = False):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True
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
    return responses


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



def main(
    model_name,
    template_path,
    danetqa_test_path,
    terra_test_path
):
    model, tokenizer, generation_config = load_saiga(model_name)
    danetqa_predictions = predict_danetqa(
        danetqa_test_path,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        template_path=template_path
    )
    terra_predictions = predict_terra(
        terra_test_path,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        template_path=template_path
    )


main("models/saiga_13b_lora", "internal_prompts/saiga_v2.json", "data/rsg/DaNetQA/val.jsonl", "data/rsg/TERRa/val.jsonl")
