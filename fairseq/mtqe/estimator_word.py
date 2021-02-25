import torch
import torch.nn.functional as F

from fairseq.models.lstm import LSTMEncoder


class EstimatorWord(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, input_embeding_dim, dropout=None, share_estimator=False, topk_time_step=1):
        super().__init__()
        # embed_dim = 16
        self.embed_dim = embed_dim
        self.share_estimator = share_estimator
        self.topk_time_step = topk_time_step
        self.gru = torch.nn.GRU(input_embeding_dim, int(embed_dim), dropout=dropout, bidirectional=True)
        self.gru_src = torch.nn.GRU(input_embeding_dim, int(embed_dim), dropout=dropout, bidirectional=True)
        self.gru_gap = torch.nn.GRU(2 * input_embeding_dim, int(embed_dim), dropout=dropout, bidirectional=True)
        # self.gru = LSTMEncoder(embed_dim=524, hidden_size=int(embed_dim), dropout=dropout, num_layers=1,
        #                        bidirectional=True)
        self.fc1_word = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
        self.fc1_gap = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
        self.fc1_word_src = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2_word = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
        self.fc2_gap = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
        self.fc3_word = torch.nn.Linear(int(embed_dim), 1)
        self.fc3_gap = torch.nn.Linear(int(embed_dim), 1)
        self.fc3_word_src = torch.nn.Linear(int(embed_dim), 1)
        if self.share_estimator:
            self.fc1_she = torch.nn.Linear(int(embed_dim * 2), int(embed_dim))
            self.dropout_she = torch.nn.Dropout(dropout) if dropout is not None else None
            self.fc2_she = torch.nn.Linear(int(embed_dim), 1)

        self.ensamble = torch.nn.Linear(2, 1)
        self.reduce_dim = torch.nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, encoder_out, share_estimator=False, ensamble=False):
        if ensamble:
            return self.ensamble(encoder_out)

        output,hn = self.gru(encoder_out.transpose(0, 1))
        # x = torch.max(output, dim=0)
        x = output.topk(self.topk_time_step, dim=0)
        x = x[0].sum(dim=0, keepdim=True)/self.topk_time_step

        if share_estimator:
            x = torch.tanh(self.fc1_she(x[0]))
            if self.dropout_she is not None:
                x = self.dropout_she(x)
            x = self.fc2_she(x)
            return torch.sigmoid(x)
        else:
            x = torch.tanh(self.fc1(x[0]))
            # hn_unbind = torch.unbind(hn)
            # x = torch.tanh(self.fc1(torch.cat(hn_unbind, dim=-1)))
            # hn = hn.transpose(0, 1) # T x B x hidden -> B x T x hidden
            # hn = hn.contiguous().view([-1, int(self.embed_dim * 2)])
            # x = torch.tanh(self.fc1(hn))
            # x = torch.tanh(self.fc1(torch.squeeze(hn, dim=0)))
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.fc2(x)
            return torch.sigmoid(x)

    def max_positions(self):
        return None

    def word_forward(self, encoder_out1, encoder_out2):
        output1, hn = self.gru(encoder_out1.transpose(0, 1))
        output2, hn = self.gru(encoder_out2.transpose(0, 1))

        x1 = torch.tanh(self.fc1_word(output1))
        x2 = torch.tanh(self.fc1_word(output2))

        if self.dropout is not None:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1 = torch.tanh(self.fc3_word(x1))
        x2 = torch.tanh(self.fc3_word(x2))

        x = 0.8*x1 + 0.2*x2

        return torch.sigmoid(x)

        #
        # x = torch.cat([x1, x2], dim=-1)
        # x = torch.tanh(self.fc2_word(x))
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x = self.fc3_word(x)
        # return torch.sigmoid(x)

    def gap_forward(self, encoder_out1, encoder_out2):
        output1, hn = self.gru_gap(encoder_out1.transpose(0, 1))
        output2, hn = self.gru_gap(encoder_out2.transpose(0, 1))

        x1 = torch.tanh(self.fc1_gap(output1))
        x2 = torch.tanh(self.fc1_gap(output2))
        if self.dropout is not None:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1 = torch.tanh(self.fc3_gap(x1))
        x2 = torch.tanh(self.fc3_gap(x2))

        x = 0.8*x1 + 0.2*x2

        return torch.sigmoid(x)

    def word_forward_single(self, encoder_out1):
        output1, hn = self.gru(encoder_out1.transpose(0, 1))

        x = torch.tanh(self.fc1_word(output1))

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3_word(x)
        return torch.sigmoid(x)

    def gap_forward_single(self, encoder_out1):
        output1, hn = self.gru_gap(encoder_out1.transpose(0, 1))

        x = torch.tanh(self.fc1_gap(output1))

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3_gap(x)
        return torch.sigmoid(x)

    def src_word_forward_single(self, encoder_out1):
        output1, hn = self.gru_src(encoder_out1.transpose(0, 1))

        x = torch.tanh(self.fc1_word_src(output1))

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3_word_src(x)
        return torch.sigmoid(x)