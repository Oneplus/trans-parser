#!/usr/bin/env python
from __future__ import print_function
from collections import Counter
import argparse

n_tot, n_uas, n_las = 0, 0, 0

parser = argparse.ArgumentParser(description='Evaluation English parsing performance.')
parser.add_argument('-l', dest='labeled', action='store_true', default=False, help='evaluate LAS')
parser.add_argument('-d', dest='detailed', action='store_true', default=False, help='log details')
parser.add_argument('filename', help='The path to the filename.')
args = parser.parse_args()

dataset = open(args.filename, 'r').read().strip().split('\n\n')

sentence_details = []
token_details = []

for _i, data in enumerate(dataset):
    lines = data.split('\n')
    _tot, _uas, _las = 0, 0, 0
    for line in lines:
        tokens = line.split()
        if tokens[5] in (".", ",", ":", "''", "``"):
            continue
        if tokens[6] == tokens[8]:
            _uas += 1
            if tokens[7] == tokens[9]:
                _las += 1
            else:
                token_details.append(tokens)
        else:
            token_details.append(tokens)
        _tot += 1
    sentence_details.append((_i, _tot, _uas, _las))
    n_tot += _tot
    n_uas += _uas
    n_las += _las

if args.labeled:
    print('%.4f' % (float(n_las) / n_tot))
else:
    print('%.4f' % (float(n_uas) / n_tot))


def n_error(_d):
    return _d[1] - _d[3] if args.labeled else _d[1] - _d[2]

if args.detailed:
    print('Sentence length factors:')
    factors = Counter()
    for d in sentence_details:
        factors[len(dataset[d[0]].split('\n'))] += n_error(d)
    for k, v in sorted(factors.items(), key=lambda x: x[0], reverse=True):
        print('{}\t{}'.format(k, v))

    print('Token postag factors:')
    factors = Counter()
    for t in token_details:
        factors[t[3]] += 1
    for k, v in sorted(factors.items(), key=lambda x: x[1], reverse=True):
        print('{}\t{}'.format(k, v))

    print('Top 10 incorrect sentences:')
    sentence_details.sort(key=n_error, reverse=True)
    for i in range(10):
        d = sentence_details[i]
        print('sid={}, #tokens={}, #wrong={}'.format(d[0], d[1], n_error(d)))
        print('{}\n'.format(dataset[d[0]]))
