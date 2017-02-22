Transition-based Parser
=======================

Hope you can find everything you want and happy parsing :simple_smile:.

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

## Train on PTB

An example of the PTB data
```
```

## Train with Different Parsers

Currently, we support the following parsers in the following papers:
* d15: [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://www.aclweb.org/anthology/P/P15/P15-1033.pdf)
* b15: [Improved Transition-based Parsing by Modeling Characters instead of Words with LSTMs](http://www.aclweb.org/anthology/D/D15/D15-1041.pdf)
* k16: [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](http://www.aclweb.org/anthology/Q/Q16/Q16-1023.pdf)

You can config different parsers with `--architecture` option in the command.

## Train on Partial Trees

Training on partially annotated trees follows [Training Dependency Parsers with Partial Annotation](https://arxiv.org/abs/1609.09247).
Example of partial annotated tree:
```

```

## Train/Test w/ Beam-search