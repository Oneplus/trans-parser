#ifndef PARSER_H
#define PARSER_H

#include "corpus.h"
#include "system.h"
#include "dynet_layer/layer.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserModel {
  TransitionSystem & system;

  explicit ParserModel(TransitionSystem & system) : system(system) {}

  ~ParserModel() {}

  TransitionSystem & get_system() { return system; }

  virtual void new_graph(dynet::ComputationGraph& cg) = 0;

  virtual std::vector<dynet::Expression> get_params() = 0;
};

struct ParserState {
  virtual ~ParserState() {}

  static std::pair<unsigned, float> get_best_action(const std::vector<float>& scores,
                                                    const std::vector<unsigned>& valid_actions);

  virtual ParserState * copy() = 0;

  virtual void new_graph(dynet::ComputationGraph& cg) = 0;

  virtual void initialize(dynet::ComputationGraph& cg,
                          const InputUnits& input) = 0;

  virtual void perform_action(const unsigned& action,
                              dynet::ComputationGraph & cg,
                              const TransitionState & state) = 0;

  virtual dynet::Expression get_scores() = 0;

  virtual std::vector<dynet::Expression> get_params() = 0;
};

struct ParserStateBuilder {
  dynet::ParameterCollection & model;
  TransitionSystem & system;

  ParserStateBuilder(dynet::ParameterCollection & model,
                     TransitionSystem & system) : model(model), system(system) {}

  virtual ParserState * build() = 0;
};

#endif  //  end for PARSER_H
