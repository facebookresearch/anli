# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import MSELoss, CrossEntropyLoss

import flint.torch_util as torch_util
from tqdm import tqdm
import os
from datetime import datetime


class EmptyScheduler(object):
    def __init__(self):
        self._state_dict = dict()

    def step(self):
        pass

    def state_dict(self):
        return self._state_dict


class ResEncoder(nn.Module):
    def __init__(self, h_size=[1024, 1024, 1024], v_size=10, embd_dim=300, mlp_d=1024,
                 dropout_r=0.1, k=3, n_layers=1, num_labels=3):
        super(ResEncoder, self).__init__()
        self.Embd = nn.Embedding(v_size, embd_dim)
        self.num_labels = num_labels

        self.lstm = nn.LSTM(input_size=embd_dim, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(embd_dim + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(embd_dim + h_size[0] * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True)

        self.h_size = h_size
        self.k = k

        # self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_1 = nn.Linear(h_size[2] * 2, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, self.num_labels)

        if n_layers == 1:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        elif n_layers == 2:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        else:
            print("Error num layers")

    def init_embedding(self, embedding):
        self.Embd.weight = embedding.weight

    def forward(self, input_ids, attention_mask, labels=None):
        # if self.max_l:
        #     l1 = l1.clamp(max=self.max_l)
        #     l2 = l2.clamp(max=self.max_l)
        #     if s1.size(0) > self.max_l:
        #         s1 = s1[:self.max_l, :]
        #     if s2.size(0) > self.max_l:
        #         s2 = s2[:self.max_l, :]
        batch_l_1 = torch.sum(attention_mask, dim=1)

        # p_s1 = self.Embd(s1)
        embedding_1 = self.Embd(input_ids)

        s1_layer1_out = torch_util.auto_rnn(self.lstm, embedding_1, batch_l_1)
        # s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

        # Length truncate
        # len1 = s1_layer1_out.size(0)
        # len2 = s2_layer1_out.size(0)
        # p_s1 = p_s1[:len1, :, :]
        # p_s2 = p_s2[:len2, :, :]

        # Using high way
        s1_layer2_in = torch.cat([embedding_1, s1_layer1_out], dim=2)
        # s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = torch_util.auto_rnn(self.lstm_1, s1_layer2_in, batch_l_1)
        # s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([embedding_1, s1_layer1_out + s1_layer2_out], dim=2)
        # s2_layer3_in = torch.cat([p_s2, s2_layer1_out + s2_layer2_out], dim=2)

        s1_layer3_out = torch_util.auto_rnn(self.lstm_2, s1_layer3_in, batch_l_1)
        # s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, batch_l_1)
        # s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        # features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
        #                       torch.abs(s1_layer3_maxout - s2_layer3_maxout),
        #                       s1_layer3_maxout * s2_layer3_maxout],
        #                      dim=1)

        features = torch.cat([s1_layer3_maxout],
                             dim=1)

        logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits)


class BagOfWords(nn.Module):
    def __init__(self, v_size=10, embd_dim=300, mlp_d=1024,
                 dropout_r=0.1, n_layers=1, num_labels=3):
        super(BagOfWords, self).__init__()
        self.Embd = nn.Embedding(v_size, embd_dim)
        self.num_labels = num_labels

        # self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_1 = nn.Linear(embd_dim, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, self.num_labels)

        if n_layers == 1:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        elif n_layers == 2:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        else:
            print("Error num layers")

    def init_embedding(self, embedding):
        self.Embd.weight = embedding.weight

    def forward(self, input_ids, attention_mask, labels=None):
        # if self.max_l:
        #     l1 = l1.clamp(max=self.max_l)
        #     l2 = l2.clamp(max=self.max_l)
        #     if s1.size(0) > self.max_l:
        #         s1 = s1[:self.max_l, :]
        #     if s2.size(0) > self.max_l:
        #         s2 = s2[:self.max_l, :]
        batch_l_1 = torch.sum(attention_mask, dim=1)

        # p_s1 = self.Embd(s1)
        embedding_1 = self.Embd(input_ids)

        s1_layer3_maxout = torch_util.avg_along_time(embedding_1, batch_l_1)
        # s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        # features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
        #                       torch.abs(s1_layer3_maxout - s2_layer3_maxout),
        #                       s1_layer3_maxout * s2_layer3_maxout],
        #                      dim=1)

        features = torch.cat([s1_layer3_maxout],
                             dim=1)

        logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits)