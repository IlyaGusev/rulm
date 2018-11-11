import torch

from allennlp.common.registrable import Registrable


class Seq2SeqEncoder(torch.nn.Module, Registrable):
    def __init__(self, output_dim: int, input_dim: int):
        super().__init__()

        self._output_dim = output_dim
        self._input_dim = input_dim

    def forward(self, *inputs):
        raise NotImplementedError()

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self._input_dim

