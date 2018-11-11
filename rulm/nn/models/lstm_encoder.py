from typing import Dict, Any

from torch.nn import Dropout, Linear, LogSoftmax, LSTM, ReLU
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from rulm.nn.models.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register("lstm")
class LstmEncoder(Seq2SeqEncoder):
    def __init__(self,
                 output_dim: int,
                 input_dim: int,
                 n_layers: int,
                 lstm_hidden_size: int,
                 lstm_dropout: float=None,
                 projection_dropout: float=None):
        super().__init__(output_dim, input_dim)

        self._lstm = LSTM(self._input_dim, lstm_hidden_size, n_layers)
        if lstm_dropout:
            self._lstm_dropout = Dropout(lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x

        self._projection_linear = Linear(lstm_hidden_size, self._output_dim)
        self._projection_relu = ReLU()
        if projection_dropout:
            self._projection_dropout = Dropout(projection_dropout)
        else:
            self._projection_dropout = lambda x: x

    def forward(self, source: Dict[str, Any]):
        inputs = source["x"]
        lengths = source["lengths"]

        inputs_packed = pack(inputs, lengths) if lengths is not None else inputs
        outputs, _  = self._lstm(inputs_packed, None)
        outputs, _ = unpack(outputs)
        outputs = self._lstm_dropout(outputs)

        outputs = self._projection_linear(outputs)
        outputs = self._projection_relu(outputs)
        outputs = self._projection_dropout(outputs)

        return outputs

