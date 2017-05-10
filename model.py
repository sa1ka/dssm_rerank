#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data import Dictionary, DataIter

def normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

class CNNEncoder(nn.Module):
    def __init__(self, nemb, sent_len,
                 num_filter,
                 lhid, dropout,
                 pre_embed=None):
        super(CNNEncoder, self).__init__()
        self.num_filter = num_filter
        self.drop = nn.Dropout(dropout)
        self.sent_len = sent_len

        self.conv3 = nn.Conv2d(1, num_filter, kernel_size=(3, nemb))
        self.conv4 = nn.Conv2d(1, num_filter, kernel_size=(4, nemb))
        self.conv5 = nn.Conv2d(1, num_filter, kernel_size=(5, nemb))
        self.convs = [self.conv3, self.conv4, self.conv5]

        self.linear1 = nn.Linear(len(self.convs) * num_filter, lhid[0])
        self.linear2 = nn.Linear(lhid[0], lhid[1])
        self.linear3 = nn.Linear(lhid[1], lhid[2])
        self.linears = [self.linear1, self.linear2, self.linear3]

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        return x

    def forward(self, embed):
        embed = embed.unsqueeze(1)
        batch_size = embed.size(0)
        cnn_outputs = map(lambda e: self.conv_and_pool(embed, e), self.convs)
        x = torch.cat(cnn_outputs, 1).view(batch_size, -1)

        for linear in self.linears:
            x = self.drop(x)
            x = F.relu(linear(x))

        return x


class RNNEncoder(nn.Module):
    def __init__(self, nemb, sent_len, dropout,
                 hidden_size, num_layers, bidirectional,
                 pre_embed=None):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.drop = nn.Dropout(dropout)

        self.rnn = nn.GRU(
            input_size = nemb,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = self.bidirectional
        )

    def forward(self, embed):

        batch_size = embed.size(0)
        bi = 2 if self.bidirectional else 1
        h0 = Variable(torch.zeros(bi * self.num_layers, batch_size, self.hidden_size)).cuda()

        output, h_n = self.rnn(embed, h0)

        hidden = torch.split(h_n, 1, 0)
        hidden = map(lambda h: torch.squeeze(h, 0), hidden)
        enc = torch.cat(hidden, 1)
        return enc

class DSSM(nn.Module):
    def __init__(self, ntokens, nemb, sent_len,
                 dropout, pre_embed, encoder, enc_params):

        super(DSSM, self).__init__()
        self.encoder = encoder

        self.embed = nn.Embedding(ntokens, nemb)
        if not pre_embed is None:
            self.embed.weight = nn.Parameter(pre_embed)

        if encoder == 'RNN':
            self.encoder = RNNEncoder(
                nemb = nemb,
                sent_len = sent_len,
                hidden_size = enc_params['hidden_size'],
                num_layers = enc_params['num_layers'],
                bidirectional = enc_params['bi'],
                dropout = dropout,
                pre_embed = pre_embed
            )
        elif encoder == 'CNN':
            self.encoder = CNNEncoder(
                nemb = nemb,
                sent_len = sent_len,
                num_filter = enc_params['num_filter'],
                lhid = [512, 512, 512],
                dropout = dropout,
                pre_embed = pre_embed
            )
        elif encoder == 'BOTH':
            self.rnn = RNNEncoder(
                nemb = nemb,
                sent_len = sent_len,
                hidden_size = enc_params['hidden_size'],
                num_layers = enc_params['num_layers'],
                bidirectional = enc_params['bi'],
                dropout = dropout,
                pre_embed = pre_embed
            )
            self.cnn = CNNEncoder(
                nemb = nemb,
                sent_len = sent_len,
                num_filter = enc_params['num_filter'],
                lhid = [512, 512, 512],
                dropout = dropout,
                pre_embed = pre_embed
            )

    def forward(self, data):
        embed = map(self.embed, data)
        if self.encoder == 'BOTH':
            c_post, c_cmnt, c_neg = map(self.cnn, embed)
            r_post, r_cmnt, r_neg = map(self.rnn, embed)
            post_enc = torch.cat((c_post, r_post), 1)
            cmnt_enc = torch.cat((c_cmnt, r_cmnt), 1)
            neg_enc = torch.cat((c_neg, r_neg), 1)
        else:
            post_enc, cmnt_enc, neg_enc = map(self.encoder, embed)

        return map(normalize, (post_enc, cmnt_enc, neg_enc))

if __name__ == '__main__':
    dic = Dictionary('./full_dataset/train.vocab')
    batch_size = 10
    seq_len = 30
    cuda = False
    data_iter = DataIter(
        corpus_path = './full_dataset/tmp.txt',
        batch_size = batch_size,
        seq_len = seq_len,
        dictionary = dic,
        cuda = cuda
    )
    ntokens = len(dic)
    enc = DSSM(
        ntokens = ntokens,
        nemb = 300,
        sent_len = seq_len,
        dropout = 0.5,
        pre_embed = None,
        encoder = 'CNN',
        enc_params = {
            'num_filter': 200,
            'lhid_size': [512, 512, 512],
        }
    )



