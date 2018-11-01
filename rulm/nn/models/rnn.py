import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from rulm.nn.config import NNConfig


class RNNModuleConfig(NNConfig):
    def __init__(self):
        self.is_binding_embeddings = True
        self.vocabulary_size = 50000
        self.embedding_size = 200
        self.embedding_dropout_p = 0.2
        self.rnn_hidden_size = 200
        self.rnn_dropout_p = 0.2
        self.n_layers = 2
        self.projection_dropout_p = 0.2


class RNNModule(nn.Module):
    def __init__(self, config: RNNModuleConfig):
        super(RNNModule, self).__init__()

        self.config = config

        vocabulary_size = self.config.vocabulary_size
        embedding_size = self.config.embedding_size
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.embedding_dropout_layer = nn.Dropout(self.config.embedding_dropout_p)

        # TODO: Not default n_layers param of LSTM because of upcoming residiual connections
        rnn_hidden_size = self.config.rnn_hidden_size
        rnn_dropout_p = self.config.rnn_dropout_p
        n_layers = self.config.n_layers
        rnns = [nn.LSTM(self.config.embedding_size, rnn_hidden_size, 1) for _ in range(n_layers)]
        dropouts = [nn.Dropout(rnn_dropout_p) for _ in range(n_layers)]
        self.rnn_layers = nn.ModuleList(rnns)
        self.dropout_layers = nn.ModuleList(dropouts)

        self.projection_layer = nn.Linear(rnn_hidden_size, embedding_size)
        self.projection_relu_layer = nn.ReLU()
        self.projection_dropout_layer = nn.Dropout(self.config.projection_dropout_p)

        self.output_dense_layer = nn.Linear(embedding_size, vocabulary_size)
        if self.config.is_binding_embeddings:
            self.output_dense_layer.weight = self.embedding_layer.weight

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, batch, lengths=None):
        inputs = self.embedding_layer(batch)
        inputs = self.embedding_dropout_layer(inputs)

        for i in range(self.config.n_layers):
            inputs_packed = pack(inputs, lengths) if lengths is not None else inputs
            outputs, _  = self.rnn_layers[i](inputs_packed, None)
            if lengths is not None:
                outputs, lengths = unpack(outputs)
            outputs = self.dropout_layers[i](outputs)
            inputs = torch.add(outputs, inputs) if i != 0 else outputs
        outputs = inputs

        outputs = self.projection_layer(outputs)
        outputs = self.projection_relu_layer(outputs)
        outputs = self.projection_dropout_layer(outputs)

        result = self.softmax(self.output_dense_layer(outputs))
        result = torch.transpose(result, 0, 1)
        result = torch.transpose(result, 1, 2)

        return result

