import pandas as pd
import numpy as np

from datasets import load_dataset
from TAPE.utils.episodes import get_episode_data

import copy

from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga


def generate(model, tokenizer, prompts, generation_config):
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
        print(tokenizer.decode(sample_input_ids, skip_special_tokens=True))
        print(sample_output)
        print()
        outputs.append(sample_output)
    return outputs


def predict_saiga_k_shots(
    model,
    tokenizer,
    generation_config,
    template_path,
    k_shots,
    questions,
    max_prompt_tokens
):
    default_conversation = Conversation.from_template(template_path)
    prompts = []
    for question in questions:
        conversation = copy.deepcopy(default_conversation)
        for user_message, bot_message in k_shots:
            conversation.add_user_message(user_message)
            conversation.add_bot_message(bot_message)
        conversation.add_user_message(question)
        prompt = conversation.get_prompt(tokenizer, max_tokens=max_prompt_tokens)
        prompts.append(prompt)
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation_config
    )


def get_data(task_name: str):
    data = load_dataset("RussianNLP/tape", f"{task_name}.episodes")
    train_data = data['train'].data.to_pandas()
    test_data = data['test'].data.to_pandas()
    return train_data, test_data


OPENBOOK_PROMPT = """Выбери правильный вариант ответа на вопрос.
Отвечай только одной буквой из набора [A, B, C, D].
Ты меня понял? (А) не понял (B) понял (C) wtf (D) ничего Я,"""


def predict(k_shots: pd.DataFrame, test_data: pd.DataFrame, task_name: str, predict_func, batch_size):
    if task_name in ['ru_worldtree', 'ru_openbook']:
        k_shots_pairs = [(OPENBOOK_PROMPT, "B")]
        for row in k_shots.to_dict(orient="records"):
            question = row["question"]
            answer = row["answer"]
            k_shots_pairs.append((question, answer))
        predictions = []
        questions = [row["question"] for row in test_data.to_dict(orient="records")]
        for batch in gen_batch(questions, batch_size=batch_size):
            batch_predictions = predict_func(
                k_shots=k_shots_pairs,
                questions=batch,
                max_prompt_tokens=1900
            )
            predictions.extend(batch_predictions)
        predictions = np.array(predictions)
    elif task_name == 'winograd':
        predictions = np.random.choice([0, 1], size=test_data.shape[0])
    elif task_name in ['chegeka', 'multiq']:
        predictions = np.random.choice(['some', 'answer'], size=test_data.shape[0])
    else:
        predictions = np.array([np.random.choice([0, 1], size=(5, )) for _ in range(test_data.shape[0])])
    return predictions


def get_predictions(task_name: str, predict_func, batch_size):
    train_data, test_data = get_data(task_name)

    full_predictions = []
    episodes = [4] + list(np.unique(np.hstack(train_data.episode.values)))
    for episode in sorted(episodes):
        k_shots = get_episode_data(train_data, episode)
        for perturbation, test in test_data.groupby('perturbation'):
            predictions = predict(k_shots, test, task_name, predict_func, batch_size)
            full_predictions.append({
                "episode": episode,
                "shot": k_shots.shape[0],
                "slice": perturbation,
                "preds": predictions
            })
    full_predictions = pd.DataFrame(full_predictions)
    return full_predictions


def main(
    model_name,
    template_path
):
    model, tokenizer, generation_config = load_saiga(model_name)

    def predict_func(k_shots, questions, max_prompt_tokens):
        return predict_saiga_k_shots(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            k_shots=k_shots,
            questions=questions,
            max_prompt_tokens=max_prompt_tokens
        )

    predictions = get_predictions("ru_openbook", predict_func, batch_size=4)
    print(predictions)
    predictions.to_json(
        'RuOpenBookQA.json',
        orient='records',
        force_ascii=False
    )


main("models/saiga_13b_lora", "internal_prompts/saiga_v2.json")
