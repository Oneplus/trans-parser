#include "archybrid.h"
#include "logging.h"
#include "corpus.h"
#include <bitset>
#include <boost/assert.hpp>

ArcHybrid::ArcHybrid(const Alphabet& map,
                     const std::string & root_string) :
  TransitionSystem(map),
  root(UINT_MAX),
  left_root(UINT_MAX),
  right_root(UINT_MAX) {

  n_actions = 1 + 2 * map.size();
  action_names.push_back("SHIFT");
  for (unsigned i = 0; i < map.size(); ++i) {
    action_names.push_back("LEFT-" + map.get(i));
    action_names.push_back("RIGHT-" + map.get(i));
    if (map.get(i) == root_string) {
      root = i;
      left_root = action_names.size() - 2;
      right_root = action_names.size() - 1;
    }
  }
  BOOST_ASSERT_MSG(root != UINT_MAX, "ROOT is not correctly set.");
  _INFO << "ArcHybrid:: show action names:";
  for (const auto& action_name : action_names) {
    _INFO << "- " << action_name;
  }
}

std::string ArcHybrid::system_name() const {
  return "archybrid";
}

std::string ArcHybrid::action_name(unsigned id) const {
  BOOST_ASSERT_MSG(id < action_names.size(), "id in illegal range");
  return action_names[id];
}

unsigned ArcHybrid::num_actions() const { return n_actions; }

unsigned ArcHybrid::num_deprels() const { return deprel_map.size(); }

void ArcHybrid::shift_unsafe(TransitionState& state) const {
  state.stack.push_back(state.buffer.back());
  state.buffer.pop_back();
}

void ArcHybrid::left_unsafe(TransitionState& state, const unsigned& deprel) const {
  unsigned hed = state.buffer.back();
  unsigned mod = state.stack.back();
  state.stack.pop_back();
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

void ArcHybrid::right_unsafe(TransitionState& state, const unsigned& deprel) const {
  unsigned mod = state.stack.back();
  state.stack.pop_back();
  unsigned hed = state.stack.back();
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

float ArcHybrid::shift_dynamic_loss_unsafe(TransitionState& state,
                                           const std::vector<unsigned>& ref_heads,
                                           const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned b = state.buffer.back();
  // The H = {s_1} U \sigma part
  // state.stack[0] = GUARD
  unsigned i_end = ((state.stack.size() > 2) ? state.stack.size() - 2 : 0);
  for (unsigned i = 1; i < i_end; ++i) {
    unsigned h = state.stack[i];
    if (ref_heads[b] == h) { c += 1.; }
  }
  if (i_end > 0) {
    unsigned h = state.stack[i_end];
    if (ref_heads[b] == h) { c += 1.; }
  }
  // The D = {s_1, s_0} U \sigma part
  // state.stack[0] = GUARD
  for (unsigned i = 1; i < state.stack.size(); ++i) {
    unsigned d = state.stack[i];
    if (ref_heads[d] == b) { c += 1.; }
  }
  shift_unsafe(state);
  return c;
}

float ArcHybrid::left_dynamic_loss_unsafe(TransitionState& state,
                                          const unsigned& deprel,
                                          const std::vector<unsigned>& ref_heads,
                                          const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned s_0 = state.stack.back();
  unsigned b = state.buffer.back();
  // The H = {s_1} U \beta part
  // state.buffer[0] = GUARD
  for (unsigned i = 1; i < state.buffer.size() - 1; ++i) {
    unsigned h = state.buffer[i];
    if (ref_heads[s_0] == h) { c += 1.; }
  }
  unsigned i_end = ((state.stack.size() > 2) ? state.stack.size() - 2 : 0);
  if (i_end > 0) {
    unsigned h = state.stack[i_end];
    if (ref_heads[s_0] == h) { c += 1.; }
  }
  // The D = {b} U \beta part
  // state.buffer[0] = GUARD
  for (unsigned i = 1; i < state.buffer.size(); ++i) {
    unsigned d = state.buffer[i];
    if (ref_heads[d] == s_0) { c += 1.; }
  }
  if (ref_heads[s_0] == b && ref_deprels[s_0] != deprel) { c += 1.; }
  left_unsafe(state, deprel);
  return c;
}

float ArcHybrid::right_dynamic_loss_unsafe(TransitionState& state,
                                           const unsigned& deprel,
                                           const std::vector<unsigned>& ref_heads,
                                           const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned s_0 = state.stack.back();
  unsigned s_1 = state.stack[state.stack.size() - 2];
  // state.buffer[0] = GUARD
  for (unsigned i = 1; i < state.buffer.size(); ++i) {
    unsigned k = state.buffer[i];
    if (ref_heads[s_0] == k) { c += 1.; }
    if (ref_heads[k] == s_0) { c += 1.; }
  }
  if (ref_heads[s_0] == s_1 && ref_deprels[s_0] != deprel) { c += 1.; }
  right_unsafe(state, deprel);
  return c;
}

void ArcHybrid::get_transition_costs(const TransitionState & state,
                                     const std::vector<unsigned>& actions,
                                     const std::vector<unsigned>& ref_heads,
                                     const std::vector<unsigned>& ref_deprels,
                                     std::vector<float> & costs) {
  float wrong_left = -1e8;
  float wrong_right = -1e8;
  costs.clear();

  for (unsigned act : actions) {
    TransitionState next_state(state);
    if (is_shift(act)) {
      costs.push_back(-shift_dynamic_loss_unsafe(next_state, ref_heads, ref_deprels));
    } else if (is_left(act)) {
      unsigned deprel = parse_label(act);
      unsigned hed = state.buffer.back(), mod = state.stack.back();
      if (ref_heads[mod] == Corpus::UNDEF_HED || (ref_heads[mod] == hed && ref_deprels[mod] == deprel)) {
        // assume that actions are unique and there is only one correct left action.
        costs.push_back(-left_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels));
      } else if (wrong_left == -1e8) {
        wrong_left = -left_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels);
        costs.push_back(wrong_left);
      } else {
        costs.push_back(wrong_left);
      }
    } else {
      unsigned deprel = parse_label(act);
      unsigned mod = state.stack.back(), hed = state.stack[state.stack.size() - 2];
      if (ref_heads[mod] == Corpus::UNDEF_HED || (ref_heads[mod] == hed && ref_deprels[mod] == deprel)) {
        costs.push_back(-right_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels));
      } else if (wrong_right == -1e8) {
        wrong_right = -right_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels);
        costs.push_back(wrong_right);
      } else {
        costs.push_back(wrong_right);
      }
    }
  }
}

void ArcHybrid::perform_action(TransitionState & state, const unsigned& action) {
  if (is_shift(action)) {
    shift_unsafe(state);
  } else if (is_left(action)) {
    left_unsafe(state, parse_label(action));
  } else {
    right_unsafe(state, parse_label(action));
  }
}

unsigned ArcHybrid::get_shift_id() { return 0; }
unsigned ArcHybrid::get_left_id(const unsigned& deprel)  { return deprel * 2 + 1; }
unsigned ArcHybrid::get_right_id(const unsigned& deprel) { return deprel * 2 + 2; }

bool ArcHybrid::is_shift(const unsigned & action) { return action == 0; }
bool ArcHybrid::is_left(const unsigned & action)  { return (action > 0 && action % 2 == 1); }
bool ArcHybrid::is_right(const unsigned & action) { return (action > 0 && action % 2 == 0); }

bool ArcHybrid::is_valid_action(const TransitionState& state, const unsigned& act) const {
  if (is_shift(act)) {
    if (state.buffer.size() == 2 && state.stack.size() > 1) { return false; }
  } else if (is_left(act)) {
    if (state.stack.size() < 2) { return false; }
    if (state.buffer.size() == 2) {
      if (state.stack.size() > 2) { return false; }
      if (act != left_root) { return false; }
    } else {
      if (act == left_root) { return false; }
    }
  } else {
    if (state.stack.size() < 3) { return false; }
    if (act == right_root) { return false; }
  }
  return true;
}

void ArcHybrid::get_valid_actions(const TransitionState& state, std::vector<unsigned>& valid_actions) {
  valid_actions.clear();
  for (unsigned a = 0; a < n_actions; ++a) {
    if (!is_valid_action(state, a)) { continue; }
    valid_actions.push_back(a);
  }
  BOOST_ASSERT_MSG(valid_actions.size() > 0, "There should be one or more valid action.");
}

unsigned ArcHybrid::parse_label(const unsigned& action) {
  BOOST_ASSERT_MSG(action > 0, "SHITF do not have label.");
  return (action - 1) / 2;
}

void ArcHybrid::get_oracle_actions(const std::vector<unsigned>& heads,
                                   const std::vector<unsigned>& deprels,
                                   std::vector<unsigned>& actions) {
  actions.clear();
  auto len = heads.size();
  std::vector<unsigned> sigma;
  std::vector<unsigned> output(len, -1);
  unsigned beta = 0;

  while (!(sigma.size() == 1 && beta == len)) {
    get_oracle_actions_onestep(heads, deprels, sigma, beta, output, actions);
  }
}

void ArcHybrid::get_oracle_actions_onestep(const std::vector<unsigned>& heads,
                                           const std::vector<unsigned>& deprels,
                                           std::vector<unsigned>& sigma,
                                           unsigned& beta,
                                           std::vector<unsigned>& output,
                                           std::vector<unsigned>& actions) {
  unsigned b = (beta < heads.size() ? beta : Corpus::BAD_HED);
  unsigned top0 = (sigma.size() > 0 ? sigma.back() : Corpus::BAD_HED);
  unsigned top1 = (sigma.size() > 1 ? sigma[sigma.size() - 2] : Corpus::BAD_HED);

  bool all_descendents_reduced = true;
  if (top0 != Corpus::BAD_HED) {
    for (unsigned i = 0; i < heads.size(); ++i) {
      if (heads[i] == top0 && output[i] != top0) { all_descendents_reduced = false; break; }
    }
  }

  if (top0 != Corpus::BAD_HED && heads[top0] == b) {
    actions.push_back(get_left_id(deprels[top0]));
    output[top0] = b;
    sigma.pop_back();
  } else if (top1 != Corpus::BAD_HED && heads[top0] == top1 && all_descendents_reduced) {
    actions.push_back(get_right_id(deprels[top0]));
    output[top0] = top1;
    sigma.pop_back();
  } else {
    BOOST_ASSERT_MSG(beta < heads.size(), "Should be more than one element in buffer.");
    actions.push_back(get_shift_id());
    sigma.push_back(beta);
    ++beta;
  }
}
