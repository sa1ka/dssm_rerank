#!/usr/bin/env python
# encoding: utf-8

import torch

from data import Dictionary, DataIter
from scipy.stats import spearmanr
from collections import defaultdict
from train import cnt_cor

def evaluate(model, data_iter):
    model.eval()
    total_cor = 0
    for d in data_iter:
        anchor, posi, neg = model(d)
        total_cor += cnt_cor(anchor, posi, neg)
    cor_rate = total_cor / float(len(data_iter))
    return cor_rate

if __name__ == '__main__':

    model_path = './params/CNN/model.pt'
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    dictionary = Dictionary('./full_dataset/train.vocab')
    print model_path

    if False:

        # test model correct rate on valid set
        valid_iter = DataIter(
            corpus_path =  './full_dataset/valid.txt',
            batch_size = 100,
            seq_len = 25,
            dictionary = dictionary,
            cuda = True
        )
        print 'cor_rate on valid: {:5.4f}'.format(evaluate(model, valid_iter))

    # test model spearman correlation with human score

    data_pathes = ['./score/0504score/', './score/0420score/', './score/0417score/', './score/0413score/']
    for data_path in data_pathes:

        print data_path
        data_iter = DataIter(
            corpus_path = data_path + 'pc_pairs.txt',
            batch_size = 1,
            seq_len = 25,
            dictionary = dictionary,
            cuda = True
        )
        with open(data_path + 'score.txt') as f:
            score = map(float, f.read().splitlines())

        model.eval()
        predict = []
        for i, d in enumerate(data_iter):
            post, cmnt, _ = model(d)
            predict.append(torch.norm(post - cmnt, 2, 1).squeeze(0).data.cpu().numpy()[0])

        with open(data_path + 'predict.txt', 'w') as f:
            for p in predict:
                f.write(str(p) + '\n')

        stat = defaultdict(lambda: [])
        for s, p in zip(score, predict):
            stat[s].append(p)

        s1 = []
        s2 = []
        for k, v in sorted(stat.iteritems()):
            s1.append(k)
            s2.append(sum(v) / len(v))

        print spearmanr(s1, s2)
