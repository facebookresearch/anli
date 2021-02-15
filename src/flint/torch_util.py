# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import functools


# def get_length_and_mask(seq):
#     len_mask = (seq != 0).long()
#     len_t = get_lengths_from_binary_sequence_mask(len_mask)
#     return len_mask, len_t


def length_truncate(seq, max_l, is_elmo=False):
    def _truncate(seq):
        if seq.size(1) > max_l:
            return seq[:, :max_l, ...]
        else:
            return seq

    if not is_elmo:
        return _truncate(seq)
    else:
        s1_elmo_embd = dict()
        s1_elmo_embd['mask'] = _truncate(seq['mask'])
        s1_elmo_embd['elmo_representations'] = []
        for e_rep in seq['elmo_representations']:
            s1_elmo_embd['elmo_representations'].append(_truncate(e_rep))
        return s1_elmo_embd


def pad_1d(seq, pad_l):
    """
    The seq is a sequence having shape [T, ..]. Note: The seq contains only one instance. This is not batched.

    :param seq:  Input sequence with shape [T, ...]
    :param pad_l: The required pad_length.
    :return:  Output sequence will have shape [Pad_L, ...]
    """
    l = seq.size(0)
    if l >= pad_l:
        return seq[:pad_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = Variable(seq.data.new(pad_l - l, *seq.size()[1:]).zero_())  # Requires_grad is False
        return torch.cat([seq, pad_seq], dim=0)


def get_state_shape(rnn: nn.RNN, batch_size, bidirectional=False):
    """
    Return the state shape of a given RNN. This is helpful when you want to create a init state for RNN.

    Example:
    c0 = h0 = Variable(src_seq_p.data.new(*get_state_shape([your rnn], 3, bidirectional)).zero_())

    :param rnn: nn.LSTM, nn.GRU or subclass of nn.RNN
    :param batch_size:
    :param bidirectional:
    :return:
    """
    if bidirectional:
        return rnn.num_layers * 2, batch_size, rnn.hidden_size
    else:
        return rnn.num_layers, batch_size, rnn.hidden_size


def pack_list_sequence(inputs, l, max_l=None, batch_first=True):
    """
    Pack a batch of Tensor into one Tensor with max_length.
    :param inputs:
    :param l:
    :param max_l: The max_length of the packed sequence.
    :param batch_first:
    :return:
    """
    batch_list = []
    max_l = max(list(l)) if not max_l else max_l
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad_1d(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1) if not batch_first \
        else torch.stack(batch_list, dim=0)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths, batch_first=True, states=None):
    """
    :param states: [rnn.num_layers, batch_size, rnn.hidden_size]
    :param inputs: Shape of the input should be [B, T, D] if batch_first else [T, B, D].
    :param lengths:  [B]
    :param batch_first:
    :return:
    """
    if not batch_first:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

        return packed_seq, reverse_indices

    else:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        if states is None:
            states = ()
        elif not isinstance(states, tuple):
            states = (states,)  # rnn.num_layers, batch_size, rnn.hidden_size

        states_lists = tuple([] for _ in states)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[i, :, :])
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

            for state_list, state in zip(states_lists, states):
                state_list.append(state[:, i, :].unsqueeze(1))

        reverse_indices = list(reverse_indices)

        s_inputs = torch.stack(s_inputs_list, dim=0)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list, batch_first=batch_first)

        r_states = tuple(torch.cat(state_list, dim=1) for state_list in states_lists)
        if len(r_states) == 1:
            r_states = r_states[0]

        return packed_seq, reverse_indices, r_states


def unpack_from_rnn_seq(packed_seq, reverse_indices, batch_first=True):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=batch_first)
    s_inputs_list = []

    if not batch_first:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)
    else:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[i, :, :].unsqueeze(0))
        return torch.cat(s_inputs_list, 0)


def reverse_indice_for_state(states, reverse_indices):
    """
    :param states: [rnn.num_layers, batch_size, rnn.hidden_size]
    :param reverse_indices: [batch_size]
    :return:
    """
    if states is None:
        states = ()
    elif not isinstance(states, tuple):
        states = (states,)  # rnn.num_layers, batch_size, rnn.hidden_size

    states_lists = tuple([] for _ in states)
    for i in reverse_indices:
        for state_list, state in zip(states_lists, states):
            state_list.append(state[:, i, :].unsqueeze(1))

    r_states = tuple(torch.cat(state_list, dim=1) for state_list in states_lists)
    if len(r_states) == 1:
        r_states = r_states[0]
    return r_states


def auto_rnn(rnn: nn.RNN, seqs, lengths, batch_first=True, init_state=None, output_last_states=False):
    batch_size = seqs.size(0) if batch_first else seqs.size(1)
    state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

    # if init_state is None:
    #     h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
    # else:
    #     h0 = init_state[0] # rnn.num_layers, batch_size, rnn.hidden_size
    #     c0 = init_state[1]

    packed_pinputs, r_index, init_state = pack_for_rnn_seq(seqs, lengths, batch_first, init_state)

    if len(init_state) == 0:
        h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
        init_state = (h0, c0)

    output, last_state = rnn(packed_pinputs, init_state)
    output = unpack_from_rnn_seq(output, r_index, batch_first)

    if not output_last_states:
        return output
    else:
        last_state = reverse_indice_for_state(last_state, r_index)
        return output, last_state


def pack_sequence_for_linear(inputs, lengths, batch_first=True):
    """
    :param inputs: [B, T, D] if batch_first
    :param lengths:  [B]
    :param batch_first:
    :return:
    """
    batch_list = []
    if batch_first:
        for i, l in enumerate(lengths):
            # print(inputs[i, :l].size())
            batch_list.append(inputs[i, :l])
        packed_sequence = torch.cat(batch_list, 0)
        # if chuck:
        #     return list(torch.chunk(packed_sequence, chuck, dim=0))
        # else:
        return packed_sequence
    else:
        raise NotImplemented()


def chucked_forward(inputs, net, chuck=None):
    if not chuck:
        return net(inputs)
    else:
        output_list = [net(chuck) for chuck in torch.chunk(inputs, chuck, dim=0)]
        return torch.cat(output_list, dim=0)


def unpack_sequence_for_linear(inputs, lengths, batch_first=True):
    batch_list = []
    max_l = max(lengths)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    if batch_first:
        start = 0
        for l in lengths:
            end = start + l
            batch_list.append(pad_1d(inputs[start:end], max_l))
            start = end
        return torch.stack(batch_list)
    else:
        raise NotImplemented()


def seq2seq_cross_entropy(logits, label, l, chuck=None, sos_truncate=True):
    """
    :param logits: [exB, V] : exB = sum(l)
    :param label: [B] : a batch of Label
    :param l: [B] : a batch of LongTensor indicating the lengths of each inputs
    :param chuck: Number of chuck to process
    :return: A loss value
    """
    packed_label = pack_sequence_for_linear(label, l)
    cross_entropy_loss = functools.partial(F.cross_entropy, size_average=False)
    total = sum(l)

    assert total == logits.size(0) or packed_label.size(0) == logits.size(0), \
        "logits length mismatch with label length."

    if chuck:
        logits_losses = 0
        for x, y in zip(torch.chunk(logits, chuck, dim=0), torch.chunk(packed_label, chuck, dim=0)):
            logits_losses += cross_entropy_loss(x, y)
        return logits_losses * (1 / total)
    else:
        return cross_entropy_loss(logits, packed_label) * (1 / total)


def max_along_time(inputs, lengths, list_in=False):
    """
    :param inputs: [B, T, D]
    :param lengths:  [B]
    :return: [B * D] max_along_time
    :param list_in:
    """
    ls = list(lengths)

    if not list_in:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)
    else:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)


def avg_along_time(inputs, lengths, list_in=False):
    """
    :param inputs: [B, T, D]
    :param lengths:  [B]
    :return: [B * D] max_along_time
    :param list_in:
    """
    ls = list(lengths)

    if not list_in:
        b_seq_avg_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_avg = seq_i.mean(dim=0)
            seq_i_avg = seq_i_avg.squeeze()
            b_seq_avg_list.append(seq_i_avg)

        return torch.stack(b_seq_avg_list)
    else:
        b_seq_avg_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i]
            seq_i_avg, _ = seq_i.mean(dim=0)
            seq_i_avg = seq_i_avg.squeeze()
            b_seq_avg_list.append(seq_i_avg)

        return torch.stack(b_seq_avg_list)


# def length_truncate(inputs, lengths, max_len):
#     """
#     :param inputs: [B, T]
#     :param lengths: [B]
#     :param max_len: int
#     :return: [B, T]
#     """
#     max_l = max(1, max_len)
#     max_s1_l = min(max(lengths), max_l)
#     lengths = lengths.clamp(min=1, max=max_s1_l)
#     if inputs.size(1) > max_s1_l:
#         inputs = inputs[:, :max_s1_l]
#
#     return inputs, lengths, max_s1_l


def get_reverse_indices(indices, lengths):
    r_indices = indices.data.new(indices.size()).fill_(0)
    batch_size = indices.size(0)
    for i in range(int(batch_size)):
        b_ind = indices[i]
        b_l = lengths[i]
        for k, ind in enumerate(b_ind):
            if k >= b_l:
                break
            r_indices[i, int(ind)] = k
    return r_indices


def index_ordering(inputs, lengths, indices, pad_value=0):
    """
    :param inputs: [B, T, ~]
    :param lengths: [B]
    :param indices: [B, T]
    :return:
    """
    batch_size = inputs.size(0)
    ordered_out_list = []
    for i in range(int(batch_size)):
        b_input = inputs[i]
        b_l = lengths[i]
        b_ind = indices[i]
        b_out = b_input[b_ind]
        if b_out.size(0) > b_l:
            b_out[b_l:] = pad_value
        ordered_out_list.append(b_out)

    outs = torch.stack(ordered_out_list, dim=0)
    return outs


def start_and_end_token_handling(inputs, lengths, sos_index=1, eos_index=2, pad_index=0,
                                 op=None):
    """
    :param inputs: [B, T]
    :param lengths: [B]
    :param sos_index:
    :param eos_index:
    :param pad_index:
    :return:
    """
    batch_size = inputs.size(0)

    if not op:
        return inputs, lengths
    elif op == 'rm_start':
        inputs = torch.cat([inputs[:, 1:], Variable(inputs.data.new(batch_size, 1).zero_())], dim=1)
        return inputs, lengths - 1
    elif op == 'rm_end':
        for i in range(batch_size):
            pass
            # Potential problems!?
            # inputs[i, lengths[i] - 1] = pad_index
        return inputs, lengths - 1
    elif op == 'rm_both':
        for i in range(batch_size):
            pass
            # Potential problems!?
            # inputs[i, lengths[i] - 1] = pad_index
        inputs = torch.cat([inputs[:, 1:], Variable(inputs.data.new(batch_size, 1).zero_())], dim=1)
        return inputs, lengths - 2


def seq2seq_att(mems, lengths, state, att_net=None):
    """
    :param mems: [B, T, D_mem] This are the memories.
                    I call memory for this variable because I think attention is just like read something and then
                    make alignments with your memories.
                    This memory here is usually the input hidden state of the encoder.

    :param lengths: [B]
    :param state: [B, D_state]
                    I call state for this variable because it's the state I percepts at this time step.

    :param att_net: This is the attention network that will be used to calculate the alignment score between
                    state and memories.
                    input of the att_net is mems and state with shape:
                        mems: [exB, D_mem]
                        state: [exB, D_state]
                    return of the att_net is [exB, 1]

                    So any function that map a vector to a scalar could work.

    :return: [B, D_result]
    """

    d_state = state.size(1)

    if not att_net:
        return state
    else:
        batch_list_mems = []
        batch_list_state = []
        for i, l in enumerate(lengths):
            b_mems = mems[i, :l]  # [T, D_mem]
            batch_list_mems.append(b_mems)

            b_state = state[i].expand(b_mems.size(0), d_state)  # [T, D_state]
            batch_list_state.append(b_state)

        packed_sequence_mems = torch.cat(batch_list_mems, 0)  # [sum(l), D_mem]
        packed_sequence_state = torch.cat(batch_list_state, 0)  # [sum(l), D_state]

        align_score = att_net(packed_sequence_mems, packed_sequence_state)  # [sum(l), 1]

        # The score grouped as [(a1, a2, a3), (a1, a2), (a1, a2, a3, a4)].
        # aligned_seq = packed_sequence_mems * align_score

        start = 0
        result_list = []
        for i, l in enumerate(lengths):
            end = start + l

            b_mems = packed_sequence_mems[start:end, :]  # [l, D_mems]
            b_score = align_score[start:end, :]  # [l, 1]

            softed_b_score = F.softmax(b_score.transpose(0, 1)).transpose(0, 1)  # [l, 1]

            weighted_sum = torch.sum(b_mems * softed_b_score, dim=0, keepdim=False)  # [D_mems]

            result_list.append(weighted_sum)

            start = end

        result = torch.stack(result_list, dim=0)

        return result

# Test something