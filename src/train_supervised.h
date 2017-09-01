#ifndef TRAIN_SUPERVISED_H
#define TRAIN_SUPERVISED_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/training.h"
#include "parser_builder.h"
#include "noisify.h"

namespace po = boost::program_options;

struct SupervisedTrainer {
  enum ORACLE_TYPE { kStatic, kDynamic };
  enum OBJECTIVE_TYPE { kCrossEntropy, kRank, kBipartieRank, kStructure };
  ORACLE_TYPE oracle_type;
  OBJECTIVE_TYPE objective_type;
  ParserStateBuilder & state_builder;
  const Noisifier& noisifier;
  float lambda_;
  float do_pretrain_iter;
  float do_explore_prob;

  static po::options_description get_options();

  SupervisedTrainer(const po::variables_map& conf,
                    const Noisifier& noisifier,
                    ParserStateBuilder & state_builder);

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

  float train_structure_full_tree(const InputUnits & input_units,
                                  const ParseUnits & parse_units,
                                  dynet::Trainer * trainer,
                                  unsigned beam_size,
                                  unsigned iter);

  float train_partial_tree(const InputUnits& input_units,
                           const ParseUnits& parse_units,
                           dynet::Trainer* trainer,
                           unsigned iter);

  void add_loss_one_step(dynet::Expression & score_expr,
                         const unsigned & best_gold_action,
                         const unsigned & worst_gold_action,
                         const unsigned & best_non_gold_action,
                         std::vector<dynet::Expression> & loss);
};

#endif  //  end for TRAIN_SUPERVISED_H