import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize(examples, tokenizer):
    outputs = tokenizer(
        examples["text"],
        truncation=False,
        max_length=None,
        padding=False
    )
    return outputs


def group(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Padding for the last example is handled in data collator
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def calc_blocks(
    dataset_path,
    tokenizer_path,
    block_size,
    streaming
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = load_dataset(dataset_path, streaming=streaming)
    datasets = datasets.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    ).map(
        lambda x: group(x, block_size),
        batched=True
    )

    train_dataset = datasets["train"]
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
