#!/usr/bin/env python
import sys
import argparse

parser = argparse.ArgumentParser(description='Evaluation English parsing performance.')
parser.add_argument('-l', dest='labeled', action='store_true', default=False, help='evaluate LAS')
parser.add_argument('filename', help='The path to the filename.')
args = parser.parse_args()

n_corr, n_uas, n_las = 0, 0, 0

for data in open(args.filename, 'r').read().strip().split('\n\n'):
    lines = data.split('\n')
    for line in lines:
        tokens = line.split()
        if tokens[5] in ("PU"):
            continue
        if tokens[6] == tokens[8]:
            n_uas += 1
            if tokens[7] == tokens[9]:
                n_las += 1
        n_corr += 1

if args.labeled:
    print '%.4f' % (float(n_las) / n_corr)
else:
    print '%.4f' % (float(n_uas) / n_corr)
