#ifndef TRAIN_SUPERVISED_ENSEMBLE_STATIC_H
#define TRAIN_SUPERVISED_ENSEMBLE_STATIC_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/training.h"
#include "parser_builder.h"
#include "noisify.h"

struct SupervisedEnsembleStaticTrainer {
  ParserStateBuilder & state_builder;
  const Noisifier& noisifier;
  float lambda_;
  float epsilon;
  unsigned n_pretrained;

  SupervisedEnsembleStaticTrainer(const po::variables_map& conf,
                                  const Noisifier& noisifier,
                                  ParserStateBuilder & state_builder);

  /* Code for supervised pretraining. */
  void train(const po::variables_map& conf,
             CorpusWithActions & corpus,
             const std::string& name,
             const std::string& output,
             bool allow_nonprojective);

  float train_full_tree(const InputUnits& input_units,
                        const ParseUnits& parse_units,
                        const ActionUnits & action_units,
                        dynet::Trainer* trainer);

  void add_loss_one_step(dynet::Expression & score_expr,
                         const std::vector<unsigned> & valid_actions,
                         const std::vector<float> & probs,
                         std::vector<dynet::Expression> & loss);
};

#endif  //  end for TRAIN_SUPERVISED_ENSEMBLE_STATIC_H