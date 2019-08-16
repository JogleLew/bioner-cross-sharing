import collections
from itertools import repeat
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def prepare_rnn_seq(rnn_input, lengths, hx=None, masks=None, batch_first=False):
    '''

    Args:
        rnn_input: [seq_len, batch, input_size]: tensor containing the features of the input sequence.
        lengths: [batch]: tensor containing the lengthes of the input sequence
        hx: [num_layers * num_directions, batch, hidden_size]: tensor containing the initial hidden state for each element in the batch.
        masks: [seq_len, batch]: tensor containing the mask for each element in the batch.
        batch_first: If True, then the input and output tensors are provided as [batch, seq_len, feature].

    Returns:

    '''
    def check_decreasing(lengths):
        lens, order = torch.sort(lengths, dim=0, descending=True)
        if torch.ne(lens, lengths).sum() == 0:
            return None
        else:
            _, rev_order = torch.sort(order)
            return lens, Variable(order), Variable(rev_order)

    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
        rev_order = None
    else:
        lens, order, rev_order = check_res
        batch_dim = 0 if batch_first else 1
        rnn_input = rnn_input.index_select(batch_dim, order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, order)
                cx = cx.index_select(1, order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, order)

    lens = lens.tolist()
    seq = rnn_utils.pack_padded_sequence(rnn_input, lens, batch_first=batch_first)
    if masks is not None:
        if batch_first:
            masks = masks[:, :lens[0]]
        else:
            masks = masks[:lens[0]]
    return seq, hx, rev_order, masks


def recover_rnn_seq(seq, rev_order, hx=None, batch_first=False):
    output, _ = rnn_utils.pad_packed_sequence(seq, batch_first=batch_first)
    if rev_order is not None:
        batch_dim = 0 if batch_first else 1
        output = output.index_select(batch_dim, rev_order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, rev_order)
                cx = cx.index_select(1, rev_order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, rev_order)
    return output, hx

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

def onehot(idxs, length):
    # idxs: [batch, 1] -> LongTensor
    # length: 1 -> Int
    # output: [batch, length] -> FloatTensor
    batch_size = idxs.size()[0]
    one_hot = torch.zeros(batch_size, length)
    if type(idxs) == torch.autograd.variable.Variable:
        idxs = idxs.data
    if type(idxs) == torch.cuda.LongTensor:
        one_hot = one_hot.cuda()
    one_hot = one_hot.scatter_(1, idxs, 1.)
    return one_hot
