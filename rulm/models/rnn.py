import torch
import torch.nn as nn

from rulm.nnconfig import NNConfig


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

    def forward(self, batch):
        inputs = self.embedding_layer(batch)
        inputs = self.embedding_dropout_layer(inputs)

        for i in range(self.config.n_layers):
            outputs, _  = self.rnn_layers[i * 2](inputs)
            outputs = self.rnn_layers[i * 2 + 1](outputs)
            inputs = torch.add(outputs, inputs) if i != 0 else outputs
        outputs = inputs

        outputs = self.projection_layer(outputs)
        outputs = self.projection_relu_layer(outputs)
        outputs = self.projection_dropout_layer(outputs)

        result = self.softmax(self.output_dense_layer(outputs))
        result = torch.transpose(result, 0, 1)
        result = torch.transpose(result, 1, 2)

        return result

