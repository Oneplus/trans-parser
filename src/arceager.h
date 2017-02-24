#ifndef ARCEAGER_H
#define ARCEAGER_H

#include "system.h"

struct ArcEager : public TransitionSystem {
  unsigned n_actions;
  std::vector<std::string> action_names;
  unsigned root;
  unsigned left_root;
  unsigned right_root;

  ArcEager(const Alphabet& deprel_map,
           const std::string & root_string);

  std::string system_name() const override;
  
  std::string action_name(unsigned id) const override;

  unsigned num_actions() const override;
  
  unsigned num_deprels() const override;

  void get_transition_costs(const TransitionState& state,
                            const std::vector<unsigned>& actions,
                            const std::vector<unsigned>& ref_heads,
                            const std::vector<unsigned>& ref_deprels,
                            std::vector<float>& rewards) override;

  void perform_action(TransitionState & state, const unsigned& action) override;
 
  bool is_valid_action(const TransitionState& state, const unsigned& act) const override;

  void get_valid_actions(const TransitionState& state,
                         std::vector<unsigned>& valid_actions) override;

  void get_oracle_actions(const std::vector<unsigned>& heads,
                          const std::vector<unsigned>& deprels,
                          std::vector<unsigned>& actions) override;

  void shift_unsafe(TransitionState& state) const;
  void left_unsafe(TransitionState& state, const unsigned& deprel) const;
  void right_unsafe(TransitionState& state, const unsigned& deprel) const;
  void reduce_unsafe(TransitionState& state) const;

  float shift_dynamic_loss_unsafe(TransitionState& state,
                                  const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels) const;

  float left_dynamic_loss_unsafe(TransitionState& state,
                                 const unsigned& deprel,
                                 const std::vector<unsigned>& heads,
                                 const std::vector<unsigned>& deprels) const;

  float right_dynamic_loss_unsafe(TransitionState& state,
                                  const unsigned& deprel,
                                  const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels) const;

  float reduce_dynamic_loss_unsafe(TransitionState& state,
                                   const std::vector<unsigned>& heads,
                                   const std::vector<unsigned>& deprels) const;

  static bool is_shift(const unsigned& action);
  static bool is_left(const unsigned& action);
  static bool is_right(const unsigned& action);
  static bool is_reduce(const unsigned& action);

  static unsigned get_shift_id();
  static unsigned get_left_id(const unsigned& deprel);
  static unsigned get_right_id(const unsigned& deprel);
  static unsigned get_reduce_id();
  static unsigned parse_label(const unsigned& action);

  void get_oracle_actions_onestep(const std::vector<unsigned>& ref_heads,
                                  const std::vector<unsigned>& ref_deprels,
                                  std::vector<unsigned>& sigma,
                                  std::vector<unsigned>& heads,
                                  unsigned& beta,
                                  const unsigned& len,
                                  std::vector<unsigned>& actions);
};

#endif  //  end for ARCEAGER_H
