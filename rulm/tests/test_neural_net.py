import unittest
import os
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.settings import REMEMBERING_EXAMPLE, ENCODER_ONLY_MODEL_PARAMS, \
    ENCODER_ONLY_SAMPLED_SOFTMAX_MODEL_PARAMS, DEFAULT_VOCAB_DIR, TRAIN_EXAMPLE, \
    TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE
from rulm.language_model import LanguageModel
from rulm.models.neural_net import NeuralNetLanguageModel


class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LanguageModel.set_seed(42)
        configs = (ENCODER_ONLY_MODEL_PARAMS, ENCODER_ONLY_SAMPLED_SOFTMAX_MODEL_PARAMS)
        cls.params_sets = [Params.from_file(config) for config in configs]

        cls.vocabularies = []
        for params in cls.params_sets:
            vocabulary_params = params.pop("vocabulary", default=Params({}))
            reader_params = params.duplicate().pop("reader", default=Params({}))
            reader = DatasetReader.from_params(reader_params)
            dataset = reader.read(REMEMBERING_EXAMPLE)
            cls.vocabularies.append(Vocabulary.from_params(vocabulary_params, instances=dataset))

        cls.train_vocabulary = Vocabulary.from_files(TRAIN_VOCAB_EXAMPLE)

        cls.sentences = []
        with open(REMEMBERING_EXAMPLE, "r", encoding="utf-8") as r:
            for line in r:
                cls.sentences.append(line.strip())

    def _test_model_predictions(self, model, reverse=False):
        for sentence in self.sentences:
            indices = model.text_to_indices(sentence)[1:-1]
            if reverse:
                indices = indices[::-1]
            sentence = " ".join([model.vocab.get_token_from_index(i) for i in indices])
            first_token = model.vocab.get_token_from_index(indices[0])
            for context_right_border in range(1, len(indices)-1):
                if context_right_border == 1 and (first_token in "Я!.?"):
                    continue
                context = indices[:context_right_border]
                text = " ".join([model.vocab.get_token_from_index(i) for i in context])
                prediction = model.sample_decoding(text, k=1)
                self.assertEqual(prediction, sentence)

    @staticmethod
    def _test_model_equality(model1: NeuralNetLanguageModel, model2: NeuralNetLanguageModel):
        old_params = list([param.detach().cpu().numpy() for param in model1.model.parameters()])
        new_params = list([param.detach().cpu().numpy() for param in model2.model.parameters()])
        for o, n in zip(old_params, new_params):
            np.testing.assert_array_almost_equal(o, n)

    def test_model_from_file(self):
        for params, vocabulary in zip(self.params_sets, self.vocabularies):
            params = params.duplicate()
            train_params = params.pop("train")
            model = LanguageModel.from_params(params, vocab=vocabulary)
            model.train(REMEMBERING_EXAMPLE, train_params)
            self._test_model_predictions(model)

    def test_reversed_model(self):
        for params, vocabulary in zip(self.params_sets, self.vocabularies):
            params = params.duplicate()
            train_params = params.pop('train')
            params["reader"]["reverse"] = True
            model_reversed = LanguageModel.from_params(params, vocab=vocabulary)
            model_reversed.train(REMEMBERING_EXAMPLE, train_params)
            self._test_model_predictions(model_reversed, reverse=True)

    def test_save_load(self):
        for params, vocabulary in zip(self.params_sets, self.vocabularies):
            with TemporaryDirectory() as dirpath:
                vocab_dir = os.path.join(dirpath, DEFAULT_VOCAB_DIR)
                os.mkdir(vocab_dir)
                vocabulary.save_to_files(vocab_dir)

                params = params.duplicate()
                train_params = params.pop('train')
                model = LanguageModel.from_params(params, vocab=vocabulary)
                model.train(REMEMBERING_EXAMPLE, train_params, dirpath)

                loaded_model = LanguageModel.load(dirpath,
                                                  params_file=ENCODER_ONLY_MODEL_PARAMS,
                                                  vocabulary_dir=vocab_dir)

                self.assertTrue(isinstance(model, NeuralNetLanguageModel))
                self.assertTrue(isinstance(loaded_model, NeuralNetLanguageModel))
                self._test_model_equality(cast(NeuralNetLanguageModel, model),
                                          cast(NeuralNetLanguageModel, loaded_model))

    def test_valid_ppl(self):
        for params in self.params_sets:
            params = params.duplicate()
            train_params = params.pop('train')
            train_params["trainer"]["num_epochs"] = 1
            train_params["iterator"]["batch_size"] = 50
            model = LanguageModel.from_params(params, vocab=self.train_vocabulary)
            metrics = model.train(TEST_EXAMPLE, train_params, valid_file_name=TEST_EXAMPLE)
            val_loss = metrics["validation_loss"]
            ppl_state = model.measure_perplexity(TEST_EXAMPLE)
            self.assertAlmostEqual(np.log(ppl_state.avg_perplexity), val_loss, places=3)
