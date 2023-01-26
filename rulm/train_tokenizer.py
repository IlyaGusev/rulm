import argparse
import random

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, Regex, decoders, trainers, processors

from rulm.util import read_jsonl


def train_tokenizer(
    dataset_path,
    train_path,
    output_dir,
    sample_rate
):
    assert train_path or dataset_path
    if train_path:
        dataset = load_dataset("rulm/jsonl_loader.py", data_files={"train": [train_path]}, streaming=True)["train"]
    elif dataset_path:
        dataset = load_dataset(dataset_path, streaming=True)["train"]

    def read_texts():
        for r in dataset:
            if random.random() < sample_rate:
                yield r["text"]

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Strip()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Digits(individual_digits=True)])
    tokenizer.decoder = decoders.Metaspace()

    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]
    trainer = trainers.UnigramTrainer(vocab_size=32768, special_tokens=special_tokens, unk_token="<unk>")
    tokenizer.train_from_iterator(read_texts(), trainer=trainer)

    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    sep_token_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s>:0 $A:0 </s>:0",
        pair="<s>:0 $A:0 <sep>:0 $B:1 </s>:1",
        special_tokens=[("<sep>", sep_token_id), ("<s>", bos_token_id), ("</s>", eos_token_id)],
    )
    encoding = tokenizer.encode("Привет! Как дела? 1994 + 11 = 2005")
    print(encoding.tokens)
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<sep>",
        padding_side="left",
    )
    wrapped_tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()
    train_tokenizer(**vars(args))
