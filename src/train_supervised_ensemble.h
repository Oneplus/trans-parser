#ifndef TRAIN_SUPERVISED_ENSEMBLE_H
#define TRAIN_SUPERVISED_ENSEMBLE_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/training.h"
#include "parser_builder.h"
#include "noisify.h"

namespace po = boost::program_options;

struct SupervisedEnsembleTrainer {
  enum ROLLIN_POLICY_TYPE { kExpert, kEpsilonGreedy };
  enum OBJECTIVE_TYPE { kCrossEntropy, kSparseCrossEntropy };
  ROLLIN_POLICY_TYPE rollin_type;
  OBJECTIVE_TYPE objective_type;
  ParserStateBuilder & state_builder;
  std::vector<ParserStateBuilder *>& pretrained_state_builders;
  const Noisifier& noisifier;
  float lambda_;
  float epsilon;
  unsigned n_pretrained;

  static po::options_description get_options();

  SupervisedEnsembleTrainer(const po::variables_map& conf,
                            const Noisifier& noisifier,
                            ParserStateBuilder & state_builder,
                            std::vector<ParserStateBuilder *>& pretrained_state_builders);

  /* Code for supervised pretraining. */
  void train(const po::variables_map& conf,
             Corpus& corpus,
             const std::string& name,
             const std::string& output,
             bool allow_nonprojective,
             bool allow_partial_tree);

  float train_full_tree(const InputUnits& input_units,
                        const ParseUnits& parse_units,
                        dynet::Trainer* trainer, 
                        unsigned iter);

  void add_loss_one_step(dynet::expr::Expression & score_expr,
                         const std::vector<unsigned> & valid_actions,
                         const std::vector<float> & probs,
                         std::vector<dynet::expr::Expression> & loss);
};

#endif  //  end for TRAIN_SUPERVISED_H