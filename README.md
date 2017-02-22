Transition-based Parser
=======================

Hope you can find everything you want and happy parsing :smile:.

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

Currently, we support the following parsers in the following papers:
* d15: [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://www.aclweb.org/anthology/P/P15/P15-1033.pdf)
* b15: [Improved Transition-based Parsing by Modeling Characters instead of Words with LSTMs](http://www.aclweb.org/anthology/D/D15/D15-1041.pdf)
* k16: [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](http://www.aclweb.org/anthology/Q/Q16/Q16-1023.pdf)

You can config different parsers with `--architecture` option in the command.

#### ROOT
Special constrains on parsing action is adopted to make sure the output tree has only one root word with root dependency relation.
`--root` is used to specify the name relation. Dummy root token is positioned at according to [Going to the roots of dependency parsing](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00132).

#### Dynamic Oracles
Dynamic oracles outputs the optimal action at non-canonical states (states that are not on the oracle transition sequence).
* ArcStandard: [A tabular method for dynamic oracles in transition-based parsing](https://transacl.org/ojs/index.php/tacl/article/view/302/38)
* {ArcHybrid|ArcEager}: [Training deterministic parsers with non-deterministic oracles](https://www.transacl.org/ojs/index.php/tacl/article/view/145/27)

Dynamic oracles are set by `--supervised_oracle` option.

#### Noisify
Two random replacement strategies are implemented:
* *singleton*: random replace singleton during training according to [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://www.aclweb.org/anthology/P/P15/P15-1033.pdf)
* *word*: word dropout strategy according to [Deep unordered composi-tion rivals syntactic methods for text classification](https://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf).

#### Partial Tree
Training on partially annotated trees follows [Training Dependency Parsers with Partial Annotation](https://arxiv.org/abs/1609.09247) and [Constrained arc-eager dependency parsing](http://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00184#.WK3jyjvyvx4)

#### Beam-Search

Using beam during training is specified by setting `--supervised_objective`to `structure`.

## Train/Test on PTB

An example of the PTB data
```
1 This This DT DT DT 2 det _ _
2 time time NN NN NN 7 tmod _ _
3 , , , , , 7 punct _ _
4 the the DT DT DT 5 det _ _
5 firms firms NNS NNS NNS 7 nsubj _ _
6 were were VBD VBD VBD 7 cop _ _
7 ready ready JJ JJ JJ 0 root _ _
8 . . . . . 7 punct _ _
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

## Train on Partial Trees
Example of partial annotated tree:

```

```

## Train/Test w/ Beam-search

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

## Results

| Arch | Sys. | Noise | Oracle | L2 | Dev.UAS | Test.UAS | Test.LAS |
|-----|-------|------|-------|----|-----|----|----|
| D15 | arcstd |         | |

