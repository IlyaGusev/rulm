import os
from typing import List, Tuple, Generator, Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.utils.data import Dataset

from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.batch import Batch, VarBatch

class NNConfig:
    def save(self, file_name):
        assert file_name.endswith(".json")
        with open(file_name, 'w', encoding="utf-8") as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + '\n')

    def load(self, file_name):
        assert file_name.endswith(".json")
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)


class RNNModuleConfig(NNConfig):
    def __init__(self):
        self.is_binding_embeddings = True
        self.vocabulary_size = 50000
        self.embedding_size = 300
        self.embedding_dropout_p = 0.3
        self.rnn_hidden_size = 300
        self.rnn_dropout_p = 0.3
        self.n_layers = 3
        self.projection_dropout_p = 0.3


class RNNModule(nn.Module):
    def __init__(self, config: RNNModuleConfig):
        super(RNNModule, self).__init__()

        self.config = config

        vocabulary_size = self.config.vocabulary_size
        embedding_size = self.config.embedding_size
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.embedding_dropout_layer = nn.Dropout(self.config.embedding_dropout_p)

        # TODO: Consider 2 ModuleLists
        # TODO: Not default n_layers param of LSTM because of upcoming residiual connections
        rnn_hidden_size = self.config.rnn_hidden_size
        rnn_dropout_p = self.config.rnn_dropout_p
        rnns = [nn.LSTM(self.config.embedding_size, rnn_hidden_size, 1),
                nn.Dropout(rnn_dropout_p)]
        for _ in range(self.config.n_layers - 1):
            rnns.append(nn.LSTM(rnn_hidden_size, rnn_hidden_size, 1))
            rnns.append(nn.Dropout(rnn_dropout_p))
        self.rnn_layers = nn.ModuleList(rnns)

        self.projection_layer = nn.Linear(rnn_hidden_size, embedding_size)
        self.projection_relu_layer = nn.ReLU()
        self.projection_dropout_layer = nn.Dropout(self.config.projection_dropout_p)

        self.output_dense_layer = nn.Linear(embedding_size, vocabulary_size)
        if self.config.is_binding_embeddings:
            self.output_dense_layer.weight = self.embedding_layer.weight

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seqs, lengths, hidden=None):
        inputs = self.embedding_layer(input_seqs)
        inputs = self.embedding_dropout_layer(inputs)

        for i in range(self.config.n_layers):
            inputs_packed = pack(inputs, lengths)
            outputs_packed, _ = self.rnn_layers[i * 2](inputs_packed, None)
            outputs, lengths = unpack(outputs_packed)
            outputs = self.rnn_layers[i * 2 + 1](outputs)
            inputs = torch.add(outputs, inputs) if i != 0 else outputs
        outputs = inputs

        outputs = self.projection_layer(outputs)
        outputs = self.projection_relu_layer(outputs)
        outputs = self.projection_dropout_layer(outputs)

        result = self.softmax(self.output_dense_layer(outputs))
        return result

class NNLanguageModel(LanguageModel):
    def __init__(self, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False, config: NNConfig=NNConfig()):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)
        self.config = config

class RNNLanguageModel(NNLanguageModel):
    def __init__(self, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False, config: NNConfig=RNNModuleConfig()):
        NNLanguageModel.__init__(self, vocabulary, transforms, reverse, config)

        self.config.vocabulary_size = min(self.config.vocabulary_size, len(vocabulary))
        self.model = RNNModule(self.config)
        use_cuda = torch.cuda.is_available()
        self.model.cuda() if use_cuda else self.model

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)

    def train(self, inputs: Generator[List[str], Any, None], batch_size: int=64,
              max_length: int=50, report_every: int=50):
        gen = self._get_batch(inputs, batch_size, max_length)
        for step, batch in enumerate(gen):
            loss = self._process_batch(batch, self.optimizer)
            if step % 10 == 0:
                print("Step: {}, loss: {}".format(step, loss))
        self.save("model.pt")

    def train_file(self, file_name, epochs: int=20, batch_size: int=64,
                   max_length: int=50, report_every: int=50):
        assert os.path.exists(file_name)
        for epoch in range(epochs):
            print("Big epoch: {}".format(epoch))
            sentences = self._parse_file_for_train(file_name)
            self.train(sentences, batch_size=batch_size,
                       max_length=max_length, report_every=report_every)

    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 1)
        result = self.model.forward(indices, [len(indices)])
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()
        return result

    def save(self, file_name):
        torch.save(self.model, file_name)

    def load(self, file_name):
        self.model = torch.load(file_name)

    def _get_batch(self, inputs: Generator[List[str], Any, None], batch_size: int, max_length):
        batch = Batch(self.vocabulary, max_length)
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            indices.append(self.vocabulary.get_eos())
            batch.add_sentence(indices)
            if len(batch) == batch_size:
                yield VarBatch(batch)
                batch = Batch(self.vocabulary, max_length)
        if len(batch) != 0:
            yield VarBatch(batch)

    def _process_batch(self, batch, optimizer=None):
        if optimizer is not None:
            optimizer.zero_grad()

        result = self.model.forward(batch.word_indices, batch.lengths)
        result = torch.transpose(result, 0, 1)
        result = torch.transpose(result, 1, 2)
        result = torch.unsqueeze(result, 2)

        target = batch.y
        target = torch.t(target)
        target = torch.unsqueeze(target, 1)

        criterion = nn.NLLLoss()
        loss = criterion(result, target)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            optimizer.step()

        return loss.data.item()
