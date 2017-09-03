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
    --lambda 1e-6 \
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
    --lambda 1e-6 \
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
    --lambda 1e-6 \
    --noisify_method singleton \
    --optimizer_enable_eta_decay true \
    --optimizer_enable_clipping true \
    --external_eval ./script/eval_ex_enpunt.py \
    --partial true
```

## Results

**UPDATE**: 2017/09/02
