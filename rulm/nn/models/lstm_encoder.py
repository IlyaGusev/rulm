from torch.nn import Dropout, Linear, LSTM, ReLU
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

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
        self.lstm_dropout = lstm_dropout
        if lstm_dropout:
            self._lstm_dropout = Dropout(lstm_dropout)

        self._projection_linear = Linear(lstm_hidden_size, self._output_dim)
        self._projection_relu = ReLU()
        self.projection_dropout = projection_dropout
        if projection_dropout:
            self._projection_dropout = Dropout(projection_dropout)

    def forward(self, inputs, mask):
        lengths = get_lengths_from_binary_sequence_mask(mask)

        inputs_packed = pack(inputs, lengths, batch_first=True)
        outputs, _ = self._lstm(inputs_packed, None)
        outputs, _ = unpack(outputs, batch_first=True)

        if self.lstm_dropout:
            outputs = self._lstm_dropout(outputs)

        outputs = self._projection_linear(outputs)
        outputs = self._projection_relu(outputs)
        if self.projection_dropout:
            outputs = self._projection_dropout(outputs)

        return outputs

