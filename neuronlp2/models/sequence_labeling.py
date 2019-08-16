__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import ChainCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, NoCRF
from ..nn import Embedding
from ..nn import utils
from .global_attention import GlobalAttention


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for ih, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            # nn.init.xavier_uniform(ih[i:i + cell.hidden_size])
            nn.init.orthogonal(hh[i:i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(0.5)
        hh_b[l // 4:l // 2].data.fill_(0.5)


def init_rnn_cell(cell, gain=1):
    if isinstance(cell, nn.LSTM):
        init_lstm(cell, gain)
    elif isinstance(cell, nn.GRU):
        init_gru(cell, gain)
    else:
        cell.reset_parameters()


class WordAndCharEmbedding(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                 embedd_word=None, embedd_char=None, p_in=0.2, elmo=False):
        super(WordAndCharEmbedding, self).__init__()
        self.use_elmo = elmo
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.dropout_in = nn.Dropout(p=p_in)

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

        # [batch, length, word_dim]
        word = self.word_embedd(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        char_size = char.size()

        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim + char_filter] 
        # or [batch, length, 1024 + word_dim + char_filter]
        input_ = torch.cat([word, char], dim=2)
        if self.use_elmo:
            input_ = torch.cat([input_, elmo_word], dim=2)
        
        # apply dropout
        input_ = self.dropout_in(input_)
        return input_, word, char, length


class WordOnlyEmbedding(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                 embedd_word=None, embedd_char=None, p_in=0.2, elmo=False):
        super(WordOnlyEmbedding, self).__init__()
        self.use_elmo = elmo
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.dropout_in = nn.Dropout(p=p_in)

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

        # [batch, length, word_dim]
        word = self.word_embedd(input_word)

        input_ = word
        if self.use_elmo:
            input_ = torch.cat([input_, elmo_word], dim=2)
        
        # apply dropout
        input_ = self.dropout_in(input_)
        return input_, word, None, length

class EncoderConvNoEmbedd(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, elmo=False):
        super(EncoderConv, self).__init__()

        self.dropout_rnn = nn.Dropout(p_rnn)
        
        if elmo:
            in_dim = 1024 + word_dim + num_filters
        else:
            in_dim = word_dim + num_filters

        self.convs = nn.Sequential(
            nn.Conv1d(in_dim, hidden_size, 3, padding=1),
            nn.ELU(0.1),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ELU(0.1)
        )

        self.dense = None
        out_dim = hidden_size
        if tag_space:
            self.dense = nn.Linear(out_dim, tag_space)
            out_dim = tag_space
        self.dense_softmax = nn.Linear(out_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

    def _get_rnn_output(self, input_embedd, word_embedd, char_embedd, mask=None, length=None, hx=None, elmo_word=None):
        input_, word, char, length = input_embedd, word_embedd, char_embedd, length
        char_size = char.size()

        output = self.convs(input_.transpose(1, 2))
        output = output.transpose(1, 2)

        output = self.dropout_rnn(output)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = F.elu(self.dense(output))

        return output, None, mask, length

    def forward(self, input_embedd, word_embedd, char_embedd, mask=None, length=None, hx=None, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(
            input_embedd, word_embedd, char_embedd, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        return output, mask, length

    def loss(self, input_embedd, word_embedd, char_embedd, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # [batch, length, tag_space]
        output, mask, length = self.forward(
            input_embedd, word_embedd, char_embedd, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # [batch, length, num_labels]
        output = self.dense_softmax(output)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)

        if length is not None and target.size(1) != mask.size(1):
            max_len = length.max()
            target = target[:, :max_len].contiguous()

        if mask is not None:
            return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / mask.sum(), \
                   (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)).sum() / num, \
                (torch.eq(preds, target).type_as(output)).sum(), preds


class BiRecurrentConvNoEmbedd(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, elmo=False, char_level_rnn=False, char_cnn=True):
        super(BiRecurrentConvNoEmbedd, self).__init__()

        self.dropout_rnn = nn.Dropout(p_rnn)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.char_level_rnn = char_level_rnn
        if char_level_rnn:
            self.char_level_rnn = True
            self.char_rnn = RNN(char_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=p_rnn)

        word_rnn_size = word_dim
        if char_cnn:
            word_rnn_size = word_rnn_size + num_filters
        if elmo:
            word_rnn_size = word_rnn_size + 1024
        if char_level_rnn:
            word_rnn_size = word_rnn_size + hidden_size * 2
        self.rnn = RNN(word_rnn_size, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

        self.dense = None
        out_dim = hidden_size * 2
        if tag_space:
            self.dense = nn.Linear(out_dim, tag_space)
            out_dim = tag_space
        self.dense_softmax = nn.Linear(out_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

        init_rnn_cell(self.rnn)

    def _get_rnn_output(self, input_embedd, word_embedd, char_embedd, mask=None, length=None, hx=None):
        input_ = input_embedd
        word = word_embedd
        char = char_embedd
        char_size = char.size() if char_embedd is not None else None

        if self.char_level_rnn:
            # [batch, length * char_length, hidden_size * 2]
            char_rnn_result, _ = self.char_rnn(char.view(char_size[0], char_size[1] * char_size[2], -1))
            char_rnn_result_size = char_rnn_result.size()
            hidden_size = char_rnn_result_size[2] // 2
            # [batch, length, char_length, hidden_size * 2]
            char_rnn_result = char_rnn_result.contiguous().view(char_size[0], char_size[1], char_size[2], -1)
            # [batch, length, hidden_size * 2]
            char_rnn_first_and_last = torch.cat(
                    [char_rnn_result[:, :, 0, :hidden_size], char_rnn_result[:, :, -1, hidden_size:]], 
                    dim=2
            )

        if self.char_level_rnn:
            input_ = torch.cat([input_, char_rnn_first_and_last], dim=2)
        
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input_, length, hx=hx, masks=mask, batch_first=True)
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size * 2]
            output, hn = self.rnn(input_, hx=hx)
        output = self.dropout_rnn(output)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = F.elu(self.dense(output))

        return output, hn, mask, length

    def forward(self, input_embedd, word_embedd, char_embedd, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = self._get_rnn_output(input_embedd, word_embedd, char_embedd, mask=mask, length=length, hx=hx)
        return output, hn, mask, length

    def loss(self, input_embedd, word_embedd, char_embedd, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # [batch, length, tag_space]
        output, hn, mask, length = self.forward(input_embedd, word_embedd, char_embedd, mask=mask, length=length, hx=hx)
        # [batch, length, num_labels]
        output = self.dense_softmax(output)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)

        if length is not None and target.size(1) != mask.size(1):
            max_len = length.max()
            target = target[:, :max_len].contiguous()

        if mask is not None:
            return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / mask.sum(), \
                   (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)).sum() / num, \
                   (torch.eq(preds, target).type_as(output)).sum(), preds

class BiRecurrentConv(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, elmo=False, char_level_rnn=False):
        super(BiRecurrentConv, self).__init__()

        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )
        self.birecurrent_conv = BiRecurrentConvNoEmbedd(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, p_rnn=p_rnn, elmo=elmo, 
                char_level_rnn=char_level_rnn
        )

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        output, hn, mask, length = self.birecurrent_conv.forward(input_, word, char, mask=mask, length=length, hx=hx)
        return output, hn, mask, length

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        return self.birecurrent_conv.loss(input_, word, char, target, mask=mask, length=length, hx=hx, leading_symbolic=leading_symbolic)


class BiRecurrentConvNoChar(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, elmo=False, char_level_rnn=False):
        super(BiRecurrentConvNoChar, self).__init__()

        self.word_char_embedd = WordOnlyEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )
        self.birecurrent_conv = BiRecurrentConvNoEmbedd(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, p_rnn=p_rnn, elmo=elmo, 
                char_level_rnn=char_level_rnn, char_cnn=False
        )

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        output, hn, mask, length = self.birecurrent_conv.forward(input_, word, char, mask=mask, length=length, hx=hx)
        return output, hn, mask, length
    
    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        return self.birecurrent_conv.loss(input_, word, char, target, mask=mask, length=length, hx=hx, leading_symbolic=leading_symbolic)


class BiVarRecurrentConv(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiVarRecurrentConv, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn)

        self.dropout_in = None
        self.dropout_rnn = nn.Dropout2d(p_rnn)

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=(p_in, p_rnn))

    def _get_rnn_output(self, input_word, input_char, mask=None, length=None, hx=None):
        # [batch, length, word_dim]
        word = self.word_embedd(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char], dim=2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask, hx=hx)
        # apply dropout for the output of rnn
        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = F.elu(self.dense(output))

        return output, hn, mask, length


class BiRecurrentConvCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False, elmo=False):
        super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn, elmo=elmo)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRF, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRF, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRF, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()


class BiRecurrentConvCRFNoChar(BiRecurrentConvNoChar):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False, elmo=False):
        super(BiRecurrentConvCRFNoChar, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn, elmo=elmo)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRFNoChar, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRFNoChar, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = super(BiRecurrentConvCRFNoChar, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()


class BiRecurrentConvAttentionCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp'):
        super(BiRecurrentConvAttentionCRF, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn, elmo=elmo)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.attention = GlobalAttention(dim=out_dim, attn_type=attention_mode)
        self.softmax = nn.Softmax(dim=2)
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None

    def _get_rnn_attention_output(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = super(BiRecurrentConvAttentionCRF, self).forward(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = self.attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = self._get_rnn_attention_output(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = self._get_rnn_attention_output(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_attention_output(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()


class MultiTaskBiRecurrentCRF(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(MultiTaskBiRecurrentCRF, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        # init task LSTM
        self.task_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)
            exec("self.task_model%d = tmp" % i)
            exec("self.task_models.append(self.task_model%d)" % i)

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        if use_gpu:
            self.d_linear = self.d_linear.cuda()
        self.softmax_dim1 = nn.Softmax(dim=1)
        if use_gpu:
            self.softmax_dim1 = self.softmax_dim1.cuda()
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        # [task_num]
        task_one_hot = utils.onehot(
                torch.LongTensor([task_id]).view(-1, 1), 
                self.task_num
        ).long().view(-1)
        task_one_hot = torch.autograd.Variable(task_one_hot, requires_grad=False).cuda()
        # [batch, 1]
        #import pdb; pdb.set_trace()
        last_idxs = (torch.sum(mask, dim=1).long() - 1).view(-1, 1)
        batch_size, sentence_len, tag_space_size = shared_output.size()
        # [batch, length, tag_space]
        last_embedd_mask = utils.onehot(last_idxs, sentence_len).unsqueeze(2).repeat(1, 1, tag_space_size)
        last_embedd_mask = torch.autograd.Variable(last_embedd_mask)
        # [batch, tag_space]
        shared_last = torch.sum(shared_output * last_embedd_mask, dim=1).contiguous()
        shared_last = utils.grad_reverse(shared_last, 1.0)

        discriminator_out = self.discriminator(shared_last)
        adversarial_loss = discriminator_out.log() * task_one_hot.view(1, -1).repeat(batch_size, 1).float()
        adversarial_loss = -adversarial_loss.sum(dim=1).mean()

        # [batch, num_label, num_label]
        s_t = torch.bmm(torch.transpose(shared_output, 1, 2), task_output)
        s_t_square = torch.mul(s_t, s_t)
        diff_loss = torch.sqrt(torch.sum(torch.sum(s_t_square, 2), 1)).mean()

        if not task_id == 0:
            task_loss = task_loss * 0.5
        final_loss = task_loss + self.adv_loss_coef * adversarial_loss + self.diff_loss_coef * diff_loss
        return final_loss, task_loss, adversarial_loss, diff_loss

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)
        mask = task_mask

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class MultiTaskBiRecurrentCRFNoChar(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(MultiTaskBiRecurrentCRFNoChar, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordOnlyEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn, char_cnn=False)

        # init task LSTM
        self.task_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn, char_cnn=False)
            exec("self.task_model%d = tmp" % i)
            exec("self.task_models.append(self.task_model%d)" % i)

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        # [task_num]
        task_one_hot = utils.onehot(
                torch.LongTensor([task_id]).view(-1, 1), 
                self.task_num
        ).long().view(-1)
        task_one_hot = torch.autograd.Variable(task_one_hot, requires_grad=False).cuda()
        import pdb; pdb.set_trace()
        # [batch, 1]
        last_idxs = (torch.sum(mask, dim=1).long() - 1).view(-1, 1)
        batch_size, sentence_len, tag_space_size = shared_output.size()
        # [batch, length, tag_space]
        last_embedd_mask = utils.onehot(last_idxs, sentence_len).unsqueeze(2).repeat(1, 1, tag_space_size)
        last_embedd_mask = torch.autograd.Variable(last_embedd_mask)
        # [batch, tag_space]
        shared_last = torch.sum(shared_output * last_embedd_mask, dim=1).contiguous()
        shared_last = utils.grad_reverse(shared_last, 1.0)

        discriminator_out = self.discriminator(shared_last)
        adversarial_loss = discriminator_out.log() * task_one_hot.view(1, -1).repeat(batch_size, 1).float()
        adversarial_loss = -adversarial_loss.sum(dim=1).mean()

        # [batch, num_label, num_label]
        s_t = torch.bmm(torch.transpose(shared_output, 1, 2), task_output)
        s_t_square = torch.mul(s_t, s_t)
        diff_loss = torch.sqrt(torch.sum(torch.sum(s_t_square, 2), 1)).mean()

        final_loss = task_loss + self.adv_loss_coef * adversarial_loss + self.diff_loss_coef * diff_loss
        return final_loss, task_loss, adversarial_loss, diff_loss

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)
        mask = task_mask

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class FullySharedBiRecurrentCRF(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(FullySharedBiRecurrentCRF, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        self.task_models = []
        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        mask = shared_mask
        task_length = shared_length

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class FullySharedBiRecurrentCRFNoCharCNN(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(FullySharedBiRecurrentCRFNoCharCNN, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordOnlyEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn,
                                char_cnn=False)

        self.task_models = []
        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        mask = shared_mask
        task_length = shared_length

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class FullySharedBiRecurrentNoCRF(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(FullySharedBiRecurrentNoCRF, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        self.task_models = []
        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = NoCRF(out_dim, num_labels)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = NoCRF(out_dim, num_labels_task[i])
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        task_mask = shared_mask

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = shared_output
        mask = shared_mask
        task_length = shared_length

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class PrivateOnlyBiRecurrentCRF(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(PrivateOnlyBiRecurrentCRF, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        # init task LSTM
        self.task_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)
            exec("self.task_model%d = tmp" % i)
            exec("self.task_models.append(self.task_model%d)" % i)

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        
        # init discriminator
        self.d_linear = nn.Linear(out_dim, task_num)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.discriminator = lambda x: self.softmax_dim1(self.d_linear(x))
        
        self.adv_loss_coef = adv_loss_coef
        self.diff_loss_coef = diff_loss_coef

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = task_output

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output = task_output

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space * 2]
        output = task_output
        mask = task_mask

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class PartialSharedNetwork(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', adv_loss_coef=0.05, diff_loss_coef=0.01, char_level_rnn=False):
        super(PartialSharedNetwork, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        self.private_models = [ nn.LSTM(out_dim, out_dim // 2, num_layers=num_layers,
                             batch_first=True, bidirectional=True, dropout=p_rnn)
                             for i in range(task_num) ]

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.private_models = [x.cuda() for x in self.private_models]

        self.private_models = nn.ModuleList(self.private_models)


        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        output, hn = self.private_models[task_id](shared_output, hx=hx)
        task_mask = shared_mask

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        # concatenate [batch, length, tag_space]
        output, hn = self.private_models[task_id].forward(shared_output, hx=hx)
        task_mask = shared_mask

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        output, hn = self.private_models[task_id](shared_output, hx=hx)
        mask = shared_mask
        task_length = shared_length

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()

class LearnWhatShareNetwork(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', char_level_rnn=False, hidden_size_private=32):
        super(LearnWhatShareNetwork, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_model = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                rnn_mode, hidden_size, num_layers, num_labels, tag_space,
                                embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)

        # init task LSTM
        self.task_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size_private, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)
            exec("self.task_model%d = tmp" % i)
            exec("self.task_models.append(self.task_model%d)" % i)

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model.cuda()
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        # init gated interaction
        self.aux_to_main = nn.Linear(out_dim, out_dim)
        self.main_to_aux = nn.Linear(out_dim, out_dim)

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -1 and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_output, _, task0_mask, task0_length = self._get_rnn_attention_output(0, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task1_output, _, task1_mask, task1_length = self._get_rnn_attention_output(1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_learned_output = task0_output * torch.sigmoid(self.aux_to_main(task1_output))
        task1_learned_output = task1_output * torch.sigmoid(self.main_to_aux(task0_output))
        task_output = task0_learned_output if task_id == 0 else task1_learned_output
        task_mask = task0_mask if task_id == 0 else task1_mask
        task_length = task0_length if task_id == 0 else task1_length

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_output, _, task0_mask, task0_length = self._get_rnn_attention_output(0, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task1_output, _, task1_mask, task1_length = self._get_rnn_attention_output(1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_learned_output = task0_output * torch.sigmoid(self.aux_to_main(task1_output))
        task1_learned_output = task1_output * torch.sigmoid(self.main_to_aux(task0_output))
        task_output = task0_learned_output if task_id == 0 else task1_learned_output
        task_mask = task0_mask if task_id == 0 else task1_mask
        task_length = task0_length if task_id == 0 else task1_length

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_output, _, shared_mask, shared_length = self._get_rnn_attention_output(-1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_output, _, task0_mask, task0_length = self._get_rnn_attention_output(0, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task1_output, _, task1_mask, task1_length = self._get_rnn_attention_output(1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        task0_learned_output = task0_output * torch.sigmoid(self.aux_to_main(task1_output))
        task1_learned_output = task1_output * torch.sigmoid(self.main_to_aux(task0_output))
        task_output = task0_learned_output if task_id == 0 else task1_learned_output
        task_mask = task0_mask if task_id == 0 else task1_mask
        task_length = task0_length if task_id == 0 else task1_length

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)
        mask = task_mask

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()


class LearnWhatShareTestNetwork(nn.Module):
    def __init__(self, task_num, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, num_labels_task=None, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False,
                 elmo=False, attention_mode='mlp', char_level_rnn=False, hidden_size_private=32):
        super(LearnWhatShareTestNetwork, self).__init__()
        use_gpu = torch.cuda.is_available()

        if rnn_mode == 'CNN':
            ENC = EncoderConvNoEmbedd
        else:
            ENC = BiRecurrentConvNoEmbedd

        # init word and char embedd
        self.word_char_embedd = WordAndCharEmbedding(
                word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, 
                embedd_word=embedd_word, embedd_char=embedd_char, p_in=p_in, elmo=elmo
        )

        # init shared LSTM
        self.shared_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size_private, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)
            exec("self.shared_model%d = tmp" % i)
            exec("self.shared_models.append(self.shared_model%d)" % i)

        # init task LSTM
        self.task_models = []
        for i in range(task_num):
            tmp = ENC(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, tag_space, embedd_word, embedd_char, p_in, p_rnn, elmo, char_level_rnn=char_level_rnn)
            exec("self.task_model%d = tmp" % i)
            exec("self.task_models.append(self.task_model%d)" % i)

        if use_gpu:
            self.word_char_embedd.cuda()
            self.shared_model = [x.cuda() for x in self.shared_models]
            self.task_models = [x.cuda() for x in self.task_models]

        out_dim = tag_space if tag_space else hidden_size * 2
        self.use_elmo = elmo

        # init attention
        self.attention_mode = attention_mode
        if attention_mode != 'none':
            self.shared_attention = GlobalAttention(dim=out_dim, attn_type=attention_mode) 
            self.task_attentions = []
            for i in range(task_num):
                tmp = GlobalAttention(dim=out_dim, attn_type=attention_mode)
                exec("self.task_attention%d = tmp" % i)
                exec("self.task_attentions.append(self.task_attention%d)" % i)
            if use_gpu:
                self.shared_attention = self.shared_attention.cuda()
                self.task_attentions = [x.cuda() for x in self.task_attentions]
        
        self.task_num = task_num
        self.softmax = nn.Softmax(dim=2)

        # init CRF
        if num_labels_task is None:
            num_labels_task = [num_labels]
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels, bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)
        else:
            self.crfs = []
            for i in range(task_num):
                tmp = ChainCRF(out_dim * 2, num_labels_task[i], bigram=bigram)
                exec("self.crf%d = tmp" % i)
                exec("self.crfs.append(self.crf%d)" % i)

        if use_gpu:
            self.crfs = [x.cuda() for x in self.crfs]

        # init gated interaction
        self.aux_to_main = nn.Linear(out_dim, out_dim)
        self.main_to_aux = nn.Linear(out_dim, out_dim)

        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        

    def _get_rnn_attention_output(self, target_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        assert target_id >= -self.task_num and target_id < self.task_num
        if self.use_elmo:
            assert elmo_word is not None

        target_model = self.task_models[target_id] if target_id >= 0 else self.shared_model[-target_id - 1]

        input_, word, char, length = self.word_char_embedd(input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)
        # output from rnn [batch, length, tag_space]
        output, hn, mask, length = target_model.forward(input_, word, char, mask=mask, length=length, hx=hx)
        if self.attention_mode == 'none':
            return output, hn, mask, length

        target_attention = self.task_attentions[target_id] if target_id >= 0 else self.shared_model
        batch_size, l, tag_space = output.size()
        assert length is not None
        attention_out, _ = target_attention(output, output, length)
        attention_out = attention_out.transpose(0, 1).contiguous() # transform to the shape of [batch, length, tag_space]
        return attention_out, hn, mask, length

    def forward(self, task_id, input_word, input_char, mask=None, length=None, hx=None, elmo_word=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_sub_output = [0, 0]
        shared_sub_mask = [0, 0]
        shared_sub_length = [0, 0]
        for i in range(len(self.shared_model)):
            shared_sub_output[i], _, shared_sub_mask[i], shared_sub_length[i] = self._get_rnn_attention_output(-i - 1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task0_learned_output = shared_sub_output[0] * torch.sigmoid(self.aux_to_main(shared_sub_output[1]))
        task1_learned_output = shared_sub_output[1] * torch.sigmoid(self.main_to_aux(shared_sub_output[0]))
        shared_output = task0_learned_output if task_id == 0 else task1_learned_output
        shared_mask = shared_sub_mask[task_id]
        shared_length = shared_sub_length[task_id]

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        # [batch, length, num_label,  num_label]
        return self.crfs[task_id](output, mask=task_mask), task_mask

    def loss(self, task_id, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_sub_output = [0, 0]
        shared_sub_mask = [0, 0]
        shared_sub_length = [0, 0]
        for i in range(len(self.shared_model)):
            shared_sub_output[i], _, shared_sub_mask[i], shared_sub_length[i] = self._get_rnn_attention_output(-i - 1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task0_learned_output = shared_sub_output[0] * torch.sigmoid(self.aux_to_main(shared_sub_output[1]))
        task1_learned_output = shared_sub_output[1] * torch.sigmoid(self.main_to_aux(shared_sub_output[0]))
        shared_output = task0_learned_output if task_id == 0 else task1_learned_output
        shared_mask = shared_sub_mask[task_id]
        shared_length = shared_sub_length[task_id]

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)

        max_len = -1
        if length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]

        if reflect is None:
            reflected_target = target
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.view(-1)).view(batch_size, sen_len)
            reflected_target = torch.autograd.Variable(reflected_target, requires_grad=True)
        task_loss = self.crfs[task_id].loss(output, reflected_target, mask=task_mask).mean()

        final_loss = task_loss 
        return final_loss, task_loss, 0.0, 0.0

    def decode(self, task_id, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0, elmo_word=None, reflect=None):
        if self.use_elmo:
            assert elmo_word is not None

        # output from rnn [batch, length, tag_space]
        shared_sub_output = [0, 0]
        shared_sub_mask = [0, 0]
        shared_sub_length = [0, 0]
        for i in range(len(self.shared_model)):
            shared_sub_output[i], _, shared_sub_mask[i], shared_sub_length[i] = self._get_rnn_attention_output(-i - 1, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task_output, _, task_mask, task_length = self._get_rnn_attention_output(task_id, input_word, input_char, mask=mask, length=length, hx=hx, elmo_word=elmo_word)

        task0_learned_output = shared_sub_output[0] * torch.sigmoid(self.aux_to_main(shared_sub_output[1]))
        task1_learned_output = shared_sub_output[1] * torch.sigmoid(self.main_to_aux(shared_sub_output[0]))
        shared_output = task0_learned_output if task_id == 0 else task1_learned_output
        shared_mask = shared_sub_mask[task_id]
        shared_length = shared_sub_length[task_id]

        # concatenate [batch, length, tag_space * 2]
        output = torch.cat([task_output, shared_output], 2)
        mask = task_mask

        if target is None:
            return self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        preds = self.crfs[task_id].decode(output, mask=mask, leading_symbolic=leading_symbolic)

        if task_length is not None:
            max_len = task_length.max()
            target = target[:, :max_len]
            preds = preds[:, :max_len]
            mask = mask[:, :max_len]

        if reflect is None:
            reflected_target = target.data
        else:
            batch_size, sen_len = target.data.size()
            reflected_target = torch.index_select(reflect, 0, target.data.contiguous().view(-1)).view(batch_size, sen_len)

        if mask is None:
            return preds, torch.eq(preds, reflected_target).float().sum()
        else:
            return preds, (torch.eq(preds, reflected_target).float() * mask.data).sum()

