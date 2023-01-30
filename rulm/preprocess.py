import argparse
import random
from itertools import chain

from datasets import load_dataset, Value, Features, Sequence
from transformers import AutoTokenizer
from tqdm import tqdm


MAX_TOKENS = 10000000
ZEROS = [0 for _ in range(MAX_TOKENS)]
ONES = [1 for _ in range(MAX_TOKENS)]


def tokenize(examples, tokenizer, position_ids):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
        return_length=True
    )
    lengths = outputs.pop("length")
    outputs["position_ids"] = [position_ids[:l] for l in lengths]
    outputs["token_type_ids"] = [ZEROS[:l] if i % 2 == 0 else ONES[:l] for i, l in enumerate(lengths)]
    return outputs


def group(examples, block_size):
    concatenated_examples = {k: list(chain(*v)) for k, v in examples.items()}
    some_key = list(examples.keys())[0]
    total_length = len(concatenated_examples[some_key])

    # Remove reminder to skip padding handling
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def preprocess(
    dataset_path,
    tokenizer_path,
    block_size,
    streaming,
    output_path
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = load_dataset(dataset_path, streaming=streaming)

    position_ids = [i % block_size for i in range(MAX_TOKENS)]
    datasets = datasets.filter(
        lambda x: random.random() < 1.0
    ).map(
        lambda x: tokenize(x, tokenizer, position_ids),
        batched=True,
        remove_columns=["text"]
    ).map(
        lambda x: group(x, block_size),
        batched=True
    ).cast(Features({
        "input_ids": Sequence(Value("uint16")),
        "position_ids": Sequence(Value("uint16")),
        "token_type_ids": Sequence(Value("bool")),
        "attention_mask": Sequence(Value("bool"))
    }))

    datasets.save_to_disk(output_path, max_shard_size="1GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--block-size", default=512, type=int)
    parser.add_argument("--streaming", action="store_true", default=False)
    args = parser.parse_args()
    preprocess(**vars(args))
