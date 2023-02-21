import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, debug=False, hidden_size=512, num_layers=1, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM', num_classes=-1):
        super(BiLtxtLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

    def forward(self, src_feats, src_lens, hidden=None):

        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens)

        if hidden is not None and self.rnn_type == 'LSTM':
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }

    def _cat_directions(self, hidden):

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            hidden = tuple([_cat(h) for h in hidden])
        else:
            hidden = _cat(hidden)

        return hidden
