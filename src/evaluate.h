#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <set>
#include "corpus.h"
#include "parser_builder.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               ParserStateBuilder & state_builder,
               const std::string & output);

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               std::vector<ParserStateBuilder *> & pretrained_state_builders,
               const std::string & output);

float beam_search(const po::variables_map & conf,
                  Corpus & corpus,
                  ParserStateBuilder & state_builder,
                  const std::string & output,
                  bool structure);

#endif  //  end for EVALUATE_H