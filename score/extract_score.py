#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict


def extract_score(filename, interval):
    post_cmnt_pairs = defaultdict(lambda: [])
    with open(filename) as f:
        lines = f.read().splitlines()
        for i in range(0, len(lines), interval):
            post = lines[i]
            for l in lines[i+1: i+interval-1]:
                _, cmnt, score = l.split('\t')
                score = int(score.strip('[]'))
                post_cmnt_pairs[(post, cmnt)].append(score)

    return post_cmnt_pairs

def merge(x, y):
    m = {}
    for key in x:
        m[key] = x[key] + y[key]
    return m


if __name__ == '__main__':
    prefix = ['rnc00', 'htl11', 'xl526', 'xyw00', 'zjz17']
    score_dir = '0504score/'
    pcp = reduce(merge, map(lambda f: extract_score(score_dir + f + '.txt', 9), prefix))
    with open(score_dir + 'pc_pairs.txt', 'w') as pc_f, open(score_dir + 'score.txt', 'w') as sf:
        for k, v in pcp.iteritems():
            pc_f.write('{0}\t=>\t{1}\t=>\t<unk>\n'.format(k[0], k[1]))
            v = map(lambda x: 0 if x == -1 else x, v)
            sf.write(str(sum(v) * 1.0 / len(v)) + '\n')

