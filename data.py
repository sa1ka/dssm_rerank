import os
import torch
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self, vocab_path):
        self.word2idx = {}
        self.idx2word = []

        with open(vocab_path) as f:
            for line in f:
                self.add_word(line.strip())
        self.pad = self['<pad>']
        self.unk = self['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.idx2word)


class DataIter(object):
    def __init__(self, corpus_path, batch_size, seq_len, dictionary, cuda=False):
        self.corpus_path = corpus_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dictionary = dictionary
        self.cuda = cuda
        self.pad = self.dictionary['<pad>']

        self.build_data()
        if self.cuda:
            self.post = self.post.cuda()
            self.cmnt = self.cmnt.cuda()
            self.neg = self.neg.cuda()

    def sent2idx(self, sent):
        '''
        the output will cut or pad to self.seq_len
        '''
        s = sent.split(' ')[0:self.seq_len]
        s = map(lambda x: self.dictionary[x], s)
        arr = s + [self.dictionary.pad] * (self.seq_len - len(s))
        return torch.LongTensor(arr)

    def build_data(self):
        with open(self.corpus_path) as f:
            lines = f.read().splitlines()
        self.nsample = len(lines)
        self.post = torch.LongTensor(self.nsample, self.seq_len)
        self.cmnt = torch.LongTensor(self.nsample, self.seq_len)
        self.neg = torch.LongTensor(self.nsample, self.seq_len)
        for i, l in enumerate(lines):
            post, cmnt, neg = map(self.sent2idx, l.split('\t=>\t'))
            self.post[i] = post
            self.cmnt[i] = cmnt
            self.neg[i] = neg

    def __iter__(self):
        for i in range(0, self.nsample, self.batch_size):
            post = Variable(self.post[i:i+self.batch_size])
            cmnt = Variable(self.cmnt[i:i+self.batch_size])
            neg = Variable(self.neg[i:i+self.batch_size])
            yield (post, cmnt, neg)

    def __len__(self):
        return self.nsample

if __name__ == '__main__':
    import time

    def time_usage(func):
        def wrapper(*args, **kwargs):
            beg_ts = time.time()
            retval = func(*args, **kwargs)
            end_ts = time.time()
            print("elapsed time: %f" % (end_ts - beg_ts))
            return retval
        return wrapper

    dictionary = Dictionary('./full_dataset/train.vocab')
    batch_size = 10
    seq_len = 30
    cuda = True
    train_iter = DataIter(
        corpus_path = './full_dataset/valid.txt',
        batch_size = batch_size,
        seq_len = seq_len,
        dictionary = dictionary,
        cuda = cuda,
    )

