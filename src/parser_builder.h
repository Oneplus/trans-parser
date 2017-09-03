#ifndef PARSER_BUILDER_H
#define PARSER_BUILDER_H

#include <iostream>
#include "corpus.h"
#include "parser.h"
#include "dynet/model.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

ParserStateBuilder * get_state_builder(const po::variables_map & conf,
                                       dynet::ParameterCollection & model,
                                       TransitionSystem & system,
                                       const Corpus & corpus,
                                       const Embeddings & pretrained);

#endif  //  end for PARSER_BUILDER_H