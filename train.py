import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim

from data import Dictionary, DataIter
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./full_dataset/',
                    help='location of the data corpus')
parser.add_argument('--encoder', type=str, default='RNN',)
parser.add_argument('--nemb', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--num_filter', type=int, default=200)
parser.add_argument('--seq_len', type=int, default=25,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./params/tmp/model.pt',
                    help='path to save the final model')
args = parser.parse_args()

print '{:=^30}'.format('all args')
for arg in vars(args):
    print ' '.join(map(str, (arg, getattr(args, arg))))

###############################################################################
# Training code
###############################################################################

def cnt_cor(anchor, posi, neg, debug=False):
    ap_dis = torch.norm(anchor - posi, 2, 1)
    an_dis = torch.norm(anchor - neg, 2, 1)
    loss = ap_dis - an_dis
    cor_list = filter(lambda x:x<0, loss.data.cpu().numpy())
    cor = len(cor_list)
    return cor


class Trainer(object):
    def __init__(self, model, criterion, ntokens, batch_size,
                 train_iter, valid_iter, test_iter=None,
                 max_epochs=50):
        self.model = model
        self.criterion = criterion
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.ntokens = ntokens
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def __train(self, lr, epoch):
        self.model.train()
        total_cor = 0
        total_loss = 0
        start_time = time.time()
        for batch, d in enumerate(self.train_iter):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            self.optim.zero_grad()

            anchor, posi, neg = self.model(d)
            loss = self.criterion(anchor, posi, neg)
            cor = cnt_cor(anchor, posi, neg)

            loss.backward()
            self.optim.step()

            total_cor += cor
            total_loss += loss.data[0]

            if batch % args.log_interval == 0 and batch > 0:
                cor_rate = total_cor / float(args.log_interval * self.batch_size)
                cur_loss = total_loss / float(args.log_interval)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | cor_rate {:5.4f} | cur_loss {:5.4f}'.format(
                    epoch, batch, len(self.train_iter)//self.batch_size, lr,
                    elapsed * 1000 / args.log_interval, cor_rate, cur_loss))
                total_cor = 0
                total_loss = 0
                start_time = time.time()

    def train(self):
        # Loop over epochs.
        lr = args.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.max_epochs+1):
                epoch_start_time = time.time()
                self.__train(lr, epoch)
                val_loss = self.evaluate(self.valid_iter)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (time.time() - epoch_start_time),))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            self.model = torch.load(f)
        if not self.test_iter is None:
            self.evaluate(self.test_iter, 'test')

    def evaluate(self, data_source, prefix='valid'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_cor = 0
        for d in data_source:
            anchor, posi, neg = self.model(d)
            cor = cnt_cor(anchor, posi, neg, True)
            total_cor += cor
        cor_rate = total_cor / float(len(data_source))
        print('| {0} cor_rate {1:5.4f}'.format(prefix, cor_rate))
        return cor_rate


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    corpus_path = args.data + '/'
    dictionary = Dictionary(corpus_path + 'train.vocab')

    train_iter = DataIter(
        corpus_path + 'train.txt',
        args.batch_size,
        args.seq_len,
        dictionary = dictionary,
        cuda = args.cuda,
    )
    valid_iter = DataIter(
        corpus_path + 'valid.txt',
        args.batch_size,
        args.seq_len,
        dictionary = dictionary,
        cuda = args.cuda,
    )

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(dictionary)
    import pickle
    pre_embed = torch.Tensor(pickle.load(open('./full_dataset/init_embed.stc', 'rb')))

    enc_params = {
        'hidden_size': args.hidden_size,
        'num_layers': args.nlayers,
        'bi': True,
        'num_filter': args.num_filter
    }

    model = model.DSSM(
        ntokens = ntokens,
        nemb = args.nemb,
        sent_len = args.seq_len,
        pre_embed = pre_embed,
        dropout = args.dropout,
        encoder = args.encoder,
        enc_params = enc_params
    )

    criterion = nn.TripletMarginLoss(margin=0.1)

    if args.cuda:
        model.cuda()
        criterion.cuda()

    trainer = Trainer(
        model = model,
        criterion = criterion,
        ntokens = ntokens,
        train_iter = train_iter,
        valid_iter = valid_iter,
        max_epochs = args.epochs,
        batch_size = args.batch_size
    )

    trainer.train()
