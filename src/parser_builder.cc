#include "parser_dyer15.h"
#include "parser_ballesteros15.h"
#include "parser_kiperwasser16.h"
#include "parser_builder.h"
#include "logging.h"

ParserStateBuilder * get_state_builder(const po::variables_map & conf,
                                       dynet::Model & model, 
                                       TransitionSystem & system,
                                       const Corpus & corpus,
                                       const Embeddings & pretrained) {
  std::string arch_name = conf["architecture"].as<std::string>();
  ParserStateBuilder * builder = nullptr;
  if (arch_name == "dyer15" || arch_name == "d15") {
    builder = new Dyer15ParserStateBuilder(conf, model, system, corpus, pretrained);
  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    builder = new Ballesteros15ParserStateBuilder(conf, model, system, corpus, pretrained);
  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    builder = new Kiperwasser16ParserStateBuilder(conf, model, system, corpus, pretrained);
  } else {
    _ERROR << "Main:: Unknown architecture name: " << arch_name;
    exit(1);
  }
  _INFO << "Main:: architecture: " << arch_name;
  return builder;
}
