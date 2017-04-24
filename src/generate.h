#ifndef GENERATE_H
#define GENERATE_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/training.h"
#include "parser_builder.h"
#include "noisify.h"

struct EnsembleStaticDataGenerator {
  enum ROLLIN_POLICY_TYPE { kExpert, kEpsilonGreedy, kBoltzmann };
  ROLLIN_POLICY_TYPE rollin_type;
  std::vector<ParserStateBuilder *>& pretrained_state_builders;
  float epsilon;
  float temperature;
  unsigned n_sample;
  unsigned n_pretrained;
  
  static po::options_description get_options();

  EnsembleStaticDataGenerator(const po::variables_map& conf,
                              std::vector<ParserStateBuilder *>& pretrained_state_builders);

  void generate(const po::variables_map& conf,
                Corpus& corpus,
                const std::string& output,
                bool allow_nonprojective);
};

#endif  //  end for GENERATE_H