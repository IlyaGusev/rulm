import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.language_model import LanguageModel


def train(model_path, train_path, val_path=None, vocabulary_path=None, config_path=None):
    vocabulary_path = vocabulary_path or os.path.join(model_path, "vocabulary")
    assert os.path.isdir(vocabulary_path), "Can't find vocab, run preprocess.py first"
    vocabulary = Vocabulary.from_files(vocabulary_path)

    config_path = config_path or os.path.join(model_path, "config.json")
    params = Params.from_file(config_path)
    train_params = params.pop("train", Params({}))
    model = LanguageModel.from_params(params, vocab=vocabulary)
    model.train(train_path, train_params, serialization_dir=model_path, valid_file_name=val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path')
    parser.add_argument('--vocabulary-path')
    parser.add_argument('--config-path')
    args = parser.parse_args()
    train(**vars(args))
