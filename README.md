Transition-based Parser
=======================

Hope you can find everything you want and happy parsing :smiley:.

## Compile

First, get the dynet and eigen3 library:
```
git submodule init
git submodule update
```
You will also need [boost](http://www.boost.org/)

Then compile:
```
mkdir build
cmake .. -DEIGEN3_INCLUDE_DIR=${YOUR_EIGEN3_PATH}
make
```
If success, you should found the executable `./bin/trans_parser`

## Concepts and Implementation Details

#### Train with Different Parsers

Currently, we support the following parsers in the corresponding papers:
* d15: [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://www.aclweb.org/anthology/P/P15/P15-1033.pdf)
* b15: [Improved Transition-based Parsing by Modeling Characters instead of Words with LSTMs](http://www.aclweb.org/anthology/D/D15/D15-1041.pdf)
* k16: [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](http://www.aclweb.org/anthology/Q/Q16/Q16-1023.pdf)

You can config different parsers with `--architecture` option in the command.

#### ROOT
Special constrains on parsing action is adopted to make sure the output tree has only one root word with root dependency relation.
`--root` is used to specify the relation name. Dummy root token is positioned at right according to [Going to the roots of dependency parsing](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00132).

#### Dynamic Oracles
Dynamic oracles outputs the optimal action at non-canonical states (states that are not on the oracle transition sequence).
* ArcStandard: [A tabular method for dynamic oracles in transition-based parsing](https://transacl.org/ojs/index.php/tacl/article/view/302/38)
* {ArcHybrid|ArcEager}: [Training deterministic parsers with non-deterministic oracles](https://www.transacl.org/ojs/index.php/tacl/article/view/145/27)

Dynamic oracles can be activated by setting `--supervised_oracle` option as `true`.

#### Noisify
Nosifying means randomly set some words as unknown word to improve the model's generalization ability.
Two random replacement strategies are implemented:
* *singleton*: random replace singleton during training according to [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://www.aclweb.org/anthology/P/P15/P15-1033.pdf)
* *word*: word dropout strategy according to [Deep unordered composition rivals syntactic methods for text classification](https://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf).

#### Partial Tree
Training on partially annotated trees generally follows [Training Dependency Parsers with Partial Annotation](https://arxiv.org/abs/1609.09247) and [Constrained arc-eager dependency parsing](http://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00184#.WK3jyjvyvx4).
The basic idea is performing constrained decoding on the partial tree to get a pseduo-oracle sequence and use it as training data.

Training on partial tree is specified by setting `--partial` option as `true`.

**WARNING**: training with partial tree on `ArcStandard` system is impossible at current status.

#### Beam-Search
Training with beam-search follows [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/abs/1603.06042).
Early stopping is used.

Training with beam-search is specified by setting `--supervised_objective`to `structure` and `--beam_size` greater than 1.

Testing with beam-search only needs to set `--beam_size` greater than 1.

## Train/test on PTB

An example of the PTB data
```
1 Ms. Ms. NNP NNP NNP 2 nn _ _
2 Haag Haag NNP NNP NNP 3 nsubj _ _
3 plays plays VBZ VBZ VBZ 0 root _ _
4 Elianti Elianti NNP NNP NNP 3 dobj _ _
5 . . . . . 3 punct _ _
```

Commands:
```
./bin/trans_parser --dynet-mem 1024 \
    --dynet-seed 1234 \
    --train \
    --architecture d15 \
    -T ./data/PTB_train_auto.conll \
    -d ./data/PTB_development_auto.conll \
    -w ./data/sskip.100.vectors.ptb_filtered \
    --lambda 1e-5 \
    --noisify_method singleton \
    --optimizer_enable_eta_decay true \
    --optimizer_enable_clipping true \
    --external_eval ./script/eval_ex_enpunt.py
```

## Train/test w/ Beam-search

Commands:
```
./bin/trans_parser --dynet-mem 1024 \
    --dynet-seed 1234 \
    --train \
    --architecture d15 \
    -T ./data/PTB_train_auto.conll \
    -d ./data/PTB_development_auto.conll \
    -w ./data/sskip.100.vectors.ptb_filtered \
    --lambda 1e-5 \
    --noisify_method singleton \
    --optimizer_enable_eta_decay true \
    --optimizer_enable_clipping true \
    --external_eval ./script/eval_ex_enpunt.py \
    --beam_size 8 \
    --supervised_objective structure
```

## Train on Partial Trees
Example of partial annotated tree:

```
1 Ms. Ms. NNP NNP NNP 2 nn _ _
2 Haag Haag NNP NNP NNP _ _ _ _
3 plays plays VBZ VBZ VBZ 0 root _ _
4 Elianti Elianti NNP NNP NNP _ _ _ _
5 . . . . . 3 punct _ _
```
The token without annotation (say Haag, Elianti in this example) is marked as `_`.

Commands:
```
./bin/trans_parser --dynet-mem 1024 \
    --dynet-seed 1234 \
    --train \
    --architecture d15 \
    -T ./data/PTB_train_auto.drop_arc_0.50.conll \
    -d ./data/PTB_development_auto.conll \
    -w ./data/sskip.100.vectors.ptb_filtered \
    --lambda 1e-5 \
    --noisify_method singleton \
    --optimizer_enable_eta_decay true \
    --optimizer_enable_clipping true \
    --external_eval ./script/eval_ex_enpunt.py \
    --partial true
```

## Results

| Arch | Sys. | Noise | Oracle | L2 | Dev.UAS | Test.UAS | Test.LAS |
|-----|-------|------|-------|----|-----|----|----|
| d15 | archyb | none | dynamic | 1e-5 | 93.11 | 93.01 | 90.84 |
| d15 | archyb | none | dynamic | 1e-6 | 93.54 | 93.38 | 91.19 |
| d15 | archyb | none | static | 1e-5 | 93.06 | 92.78 | 90.63 |
| d15 | archyb | none | static | 1e-6 | 93.41 | 92.99 | 90.86 |
| d15 | archyb | singleton | dynamic | 1e-5 | 93.13 | 92.61 | 90.46 |
| d15 | archyb | singleton | dynamic | 1e-6 | 93.55 | 93.12 | 91.02 |
| d15 | archyb | singleton | static | 1e-5 | 93.04 | 92.78 | 90.74 |
| d15 | archyb | singleton | static | 1e-6 | 93.49 | 93.08 | 90.88 |
| d15 | archyb | word | dynamic | 1e-5 | 93.14 | 92.72 | 90.63 |
| d15 | archyb | word | dynamic | 1e-6 | 93.74 | 93.09 | 90.96 |
| d15 | archyb | word | static | 1e-5 | 93.26 | 92.75 | 90.61 |
| d15 | archyb | word | static | 1e-6 | 93.45 | 93.10 | 91.05 |
| d15 | arcstd | none | dynamic | 1e-5 | 93.05 | 92.75 | 90.65 |
| d15 | arcstd | none | dynamic | 1e-6 | 93.64 | 93.49 | 91.42 |
| d15 | arcstd | none | static | 1e-5 | 93.10 | 92.59 | 90.45 |
| d15 | arcstd | none | static | 1e-6 | 93.24 | 92.93 | 90.72 |
| d15 | arcstd | singleton | dynamic | 1e-5 | 93.13 | 92.48 | 90.38 |
| d15 | arcstd | singleton | dynamic | 1e-6 | 93.71 | 93.52 | 91.43 |
| d15 | arcstd | singleton | static | 1e-5 | 93.08 | 92.70 | 90.63 |
| d15 | arcstd | singleton | static | 1e-6 | 93.39 | 93.20 | 91.09 |
| d15 | arcstd | word | dynamic | 1e-5 | 93.23 | 92.47 | 90.38 |
| d15 | arcstd | word | dynamic | 1e-6 | 93.65 | 93.46 | 91.35 |
| d15 | arcstd | word | static | 1e-5 | 93.03 | 92.80 | 90.65 |
| d15 | arcstd | word | static | 1e-6 | 93.25 | 92.90 | 90.67 |
| b15 | archyb | - | dynamic | 1e-6 | 93.49 | 93.09 | 90.94 |
| b15 | archyb | - | static | 1e-6 | 93.16 | 92.79 | 90.66 |
| b15 | arcstd | - | dynamic | 1e-6 | 93.53 | 93.26 | 91.22 |
| b15 | arcstd | - | static | 1e-6 | 93.12 | 92.97 | 90.81 |
| k16 | archyb | none | dynamic | 1e-6 | 93.13 | 92.70 | 90.55 |
| k16 | archyb | none | static | 1e-6 | 92.98 | 92.39 | 90.15 |
| k16 | archyb | singleton | dynamic | 1e-6 | 93.02 | 92.80 | 90.46 |
| k16 | archyb | singleton | static | 1e-6 | 92.83 | 92.62 | 90.45 |
| k16 | archyb | word | dynamic | 1e-6 | 93.15 | 92.91 | 90.78 |
| k16 | archyb | word | static | 1e-6 | 92.92 | 92.54 | 90.33 |
| k16 | arcstd | none | dynamic | 1e-6 | 93.24 | 92.73 | 90.48 |
| k16 | arcstd | none | static | 1e-6 | 92.91 | 92.77 | 90.46 |
| k16 | arcstd | singleton | dynamic | 1e-6 | 93.15 | 92.89 | 90.74 |
| k16 | arcstd | singleton | static | 1e-6 | 92.88 | 92.70 | 90.32 |
| k16 | arcstd | word | dynamic | 1e-6 | 93.06 | 92.93 | 90.78 |
| k16 | arcstd | word | static | 1e-6 | 92.94 | 92.35 | 90.21 |