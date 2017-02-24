#ifndef ABSTRACT_SYSTEM_H
#define ABSTRACT_SYSTEM_H

#include <vector>
#include "state.h"
#include "corpus.h"

struct TransitionState {
  static const unsigned MAX_N_WORDS = 1024;

  std::vector<unsigned> stack;
  std::vector<unsigned> buffer;
  std::vector<unsigned> heads;
  std::vector<unsigned> deprels;

  TransitionState(unsigned n);

  void initialize(const InputUnits & input);
  bool terminated() const;
};

struct TransitionSystem {
  const Alphabet& deprel_map;

  TransitionSystem(const Alphabet& map) : deprel_map(map) {}
 
  virtual std::string system_name() const = 0;

  virtual std::string action_name(unsigned id) const = 0;

  virtual unsigned num_actions() const = 0;

  virtual unsigned num_deprels() const = 0;

  virtual void get_transition_costs(const TransitionState& state,
                                    const std::vector<unsigned>& actions,
                                    const std::vector<unsigned>& ref_heads,
                                    const std::vector<unsigned>& ref_deprels,
                                    std::vector<float>& rewards) = 0;

  virtual void perform_action(TransitionState& state,
                              const unsigned& action) = 0;

  virtual bool is_valid_action(const TransitionState& state,
                               const unsigned& act) const = 0;

  virtual void get_valid_actions(const TransitionState& state,
                                 std::vector<unsigned>& valid_actions) = 0;
  
  virtual void get_oracle_actions(const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels,
                                  std::vector<unsigned>& actions) = 0;
};

#endif  //  end for SYSTEM_H
