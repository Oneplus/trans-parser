#ifndef PARSER_BUILDER_H
#define PARSER_BUILDER_H

#include <iostream>
#include "corpus.h"
#include "parser.h"
#include "dynet/model.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserStateBuilder {
  enum PARSER_ID { kDyer15, kBallesteros15, kKiperwasser16 };
  PARSER_ID parser_id;

  dynet::Model & model;
  ParserModel * parser_model;
  static po::options_description get_options();

  ParserStateBuilder(const po::variables_map & conf,
                     dynet::Model & model,
                     TransitionSystem & system,
                     const Corpus & corpus,
                     const Embeddings & pretrained);

  ParserState* build();
};
#endif  //  end for PARSER_BUILDER_H