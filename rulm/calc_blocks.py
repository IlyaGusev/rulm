import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize(element, tokenizer, block_size):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=block_size,
        return_overflowing_tokens=True,
        padding=True
    )
    return {"input_ids": outputs["input_ids"]}


def calc_blocks(
    dataset_path,
    tokenizer_path,
    block_size,
    streaming
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = load_dataset(dataset_path, streaming=streaming)
    tokenized_datasets = datasets.map(
        lambda x: tokenize(x, tokenizer, block_size),
        batched=True,
        remove_columns=["text"]
    )

    train_dataset = tokenized_datasets["train"]
    count = 0
    for example in tqdm(train_dataset):
        count += 1
    print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--block-size", default=1024, type=int)
    parser.add_argument("--streaming", action="store_true", default=False)
    args = parser.parse_args()
    calc_blocks(**vars(args))
