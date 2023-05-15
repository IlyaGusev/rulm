import time
import os
import random
import shutil
from jinja2 import Template

import fire
from tqdm import tqdm

from src.util.io import read_jsonl, write_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def to_messages(prompt):
    return [{"role": "user", "content": prompt}]


def depth_encode_prompt(task, template_path, methods_path):
    with open(methods_path) as r:
        methods = [line.strip() for line in r]
        methods = [m for m in methods if m.strip()]

    with open(template_path) as f:
        template = Template(f.read())

    return [to_messages(template.render(
        task=task,
        method=method
    ).strip()) for method in methods]


def task_only_encode_prompt(task, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return to_messages(template.render(
        task=task,
    ).strip())


def elimination_encode_prompt(task, template_path):
    with open(template_path) as f:
        template = Template(f.read())

    first_task = task
    second_task = task["previous_tasks"][-1]

    return to_messages(template.render(
        first_task=first_task,
        second_task=second_task
    ).strip())


def extend_post_process(response, original_task, method):
    if not response:
        return None
    if response["finish_reason"] == "length":
        return None
    content = response["message"]["content"].strip()
    previous_tasks = []
    if "previous_tasks" in original_task:
        previous_tasks = original_task["previous_tasks"]
    previous_tasks.append({
        "instruction": original_task["instruction"],
        "input": original_task["input"],
        "method": method
    })
    return {
        "instruction": content,
        "input": "",
        "previous_tasks": previous_tasks
    }


def check_new_task(task):
    if "оригинальное задание" in task["instruction"].lower():
        return False
    if "усложнённое задание" in task["instruction"].lower():
        return False
    if "####" in task["instruction"]:
        return False
    return True


def get_key(task):
    if "previous_tasks" in task:
        return (task["previous_tasks"][0]["instruction"], )
    return (task["instruction"], )


def process_method(
    original_tasks,
    model_name,
    decoding_args,
    method,
    probability = 1.0,
    depth_template_path = None,
    depth_methods_path = None,
    breadth_template_path = None,
    xml_template_path = None,
    json_template_path = None,
    few_shot_template_path = None
):
    assert method in ("depth", "breadth", "few_shot")
    original_tasks = [task for task in original_tasks if random.random() < probability]
    if method == "depth":
        batch = [depth_encode_prompt(task, depth_template_path, depth_methods_path) for task in original_tasks]
        for batch_part, task in zip(batch, original_tasks):
            batch_part.append(task_only_encode_prompt(task, xml_template_path))
            batch_part.append(task_only_encode_prompt(task, json_template_path))
        batch = [random.choice(batch_part) for batch_part in batch]
    elif method == "breadth":
        batch = [task_only_encode_prompt(task, breadth_template_path) for task in original_tasks]
    elif method == "few_shot":
        original_tasks = [task for task in original_tasks if task["input"]]
        batch = [task_only_encode_prompt(task, few_shot_template_path) for task in original_tasks]

    if not batch:
        return []

    results = openai_batch_completion(
        batch=batch,
        model_name=model_name,
        decoding_args=decoding_args
    )

    gen_tasks = []
    for result, original_task in zip(results, original_tasks):
        gen_tasks.append(extend_post_process(result, original_task, method=method))

    total, keep = len(gen_tasks), 0
    new_tasks = []
    for task in gen_tasks:
        if not task:
            continue
        if not check_new_task(task):
            continue
        keep += 1
        new_tasks.append(task)
    return new_tasks


def elimination_process_batch(tasks, model_name, decoding_args, template_path):
    batch = [elimination_encode_prompt(task, template_path) for task in tasks]
    results = openai_batch_completion(
        batch=batch,
        model_name=model_name,
        decoding_args=decoding_args
    )

    filtered_tasks = []
    eliminated_count = 0
    total_count = len(tasks)
    for result, task in zip(results, tasks):
        content = result["message"]["content"]
        if content.lower().startswith("да"):
            eliminated_count += 1
            continue
        filtered_tasks.append(task)
    print(f"Eliminated {eliminated_count} of {total_count} new tasks")
    return filtered_tasks


def process_batch(
    batch,
    model_name,
    decoding_args,
    depth_template_path,
    depth_methods_path,
    breadth_template_path,
    elimination_template_path,
    xml_template_path,
    json_template_path,
    few_shot_template_path
):
    new_depth_tasks = process_method(
        batch,
        model_name=model_name,
        decoding_args=decoding_args,
        method="depth",
        depth_template_path=depth_template_path,
        depth_methods_path=depth_methods_path,
        json_template_path=json_template_path,
        xml_template_path=xml_template_path
    )

    new_breadth_tasks = process_method(
        batch,
        model_name=model_name,
        decoding_args=decoding_args,
        method="breadth",
        probability=0.5,
        breadth_template_path=breadth_template_path
    )

    new_few_shot_tasks = process_method(
        batch,
        model_name=model_name,
        decoding_args=decoding_args,
        method="few_shot",
        few_shot_template_path=few_shot_template_path
    )

    new_tasks = new_depth_tasks + new_breadth_tasks + new_few_shot_tasks
    if not new_tasks:
        return []

    new_tasks = elimination_process_batch(
        new_tasks,
        model_name=model_name,
        decoding_args=decoding_args,
        template_path=elimination_template_path
    )
    return new_tasks


def improve_instructions(
    original_tasks_path,
    output_path,
    depth_template_path,
    depth_methods_path,
    breadth_template_path,
    elimination_template_path,
    xml_template_path,
    json_template_path,
    few_shot_template_path,
    model_name="gpt-3.5-turbo",
    request_batch_size=5,
    temperature=1.0,
    top_p=0.95,
    num_cpus=8,
):
    original_tasks = read_jsonl(original_tasks_path)
    print(f"Loaded {len(original_tasks)} original tasks")

    new_tasks = []
    existing_keys = set()
    if os.path.exists(output_path):
        new_tasks = read_jsonl(output_path)
        existing_keys = {get_key(task) for task in new_tasks}
        print(f"Loaded {len(new_tasks)} new tasks")

    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        top_p=top_p
    )

    is_output_printed = False
    batch = []
    for original_task in tqdm(original_tasks):
        if get_key(original_task) in existing_keys:
            print("Skip:", get_key(original_task))
            continue

        batch.append(original_task)

        if len(batch) == request_batch_size:
            new_batch_tasks = process_batch(
                batch,
                model_name=model_name,
                decoding_args=decoding_args,
                depth_template_path=depth_template_path,
                depth_methods_path=depth_methods_path,
                breadth_template_path=breadth_template_path,
                elimination_template_path=elimination_template_path,
                xml_template_path=xml_template_path,
                json_template_path=json_template_path,
                few_shot_template_path=few_shot_template_path
            )
            new_tasks.extend(new_batch_tasks)
            batch = []
            if not is_output_printed and new_batch_tasks:
                is_output_printed = True
                print(new_batch_tasks[0])

            write_jsonl(new_tasks, output_path + "_tmp")
            shutil.move(output_path + "_tmp", output_path)

    if batch:
        new_batch_tasks = process_batch(
            batch,
            model_name=model_name,
            decoding_args=decoding_args,
            depth_template_path=depth_template_path,
            depth_methods_path=depth_methods_path,
            breadth_template_path=breadth_template_path,
            elimination_template_path=elimination_template_path,
            xml_template_path=xml_template_path,
            json_template_path=json_template_path
        )
        new_tasks.extend(new_batch_tasks)
        write_jsonl(new_tasks, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(improve_instructions)
