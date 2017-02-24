#ifndef PARSER_H
#define PARSER_H

#include "layer.h"
#include "corpus.h"
#include "system.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserModel {
  TransitionSystem & system;

  ParserModel(TransitionSystem & system) : system(system) {}

  TransitionSystem & get_system() { return system; }

  virtual void new_graph(dynet::ComputationGraph& cg) = 0;
};

struct ParserState {
  static std::pair<unsigned, float> get_best_action(const std::vector<float>& scores,
                                                    const std::vector<unsigned>& valid_actions);

  virtual ParserState * copy() = 0;

  virtual void new_graph(dynet::ComputationGraph& cg) = 0;

  virtual void initialize(dynet::ComputationGraph& cg,
                          const InputUnits& input) = 0;

  virtual void perform_action(const unsigned& action,
                              dynet::ComputationGraph & cg,
                              const TransitionState & state) = 0;

  virtual dynet::expr::Expression get_scores() = 0;
};

#endif  //  end for PARSER_H
