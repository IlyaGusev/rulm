import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params

from rulm.stream_reader import LanguageModelingStreamReader
from rulm.language_model import LanguageModel


def train(model_path, train_path, val_path):
    reader = LanguageModelingStreamReader()
    dataset = reader.read(train_path)

    vocabulary_path = os.path.join(model_path, "vocabulary")
    config_path = os.path.join(model_path, "config.json")

    if os.path.isdir(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path)
    else:
        vocabulary = Vocabulary.from_instances(dataset, max_vocab_size=50000)
        vocabulary.save_to_files(vocabulary_path)

    params = Params.from_file(config_path)
    train_params = params.pop("train", Params({}))

    model = LanguageModel.from_params(params, vocab=vocabulary)
    model.train(train_path, train_params, serialization_dir=model_path, valid_file_name=val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path')
    args = parser.parse_args()
    train(args.model_path, args.train_path, args.val_path)
