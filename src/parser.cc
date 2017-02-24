#include "parser.h"
#include "dynet/expr.h"
#include "corpus.h"
#include "logging.h"
#include <vector>
#include <random>


std::pair<unsigned, float> ParserState::get_best_action(const std::vector<float>& scores,
                                                        const std::vector<unsigned>& valid_actions) {
  unsigned best_a = valid_actions[0];
  float best_score = scores[best_a];
  //! should use next valid action.
  for (unsigned i = 1; i < valid_actions.size(); ++i) {
    unsigned a = valid_actions[i];
    if (best_score < scores[a]) {
      best_a = a;
      best_score = scores[a];
    }
  }
  return std::make_pair(best_a, best_score);
}

/*
void Parser::predict(dynet::ComputationGraph& cg,
                     const InputUnits& input,
                     ParseUnits& parse) {
  new_graph(cg);

  unsigned len = input.size();
  State state(len);
  StateCheckpoint * checkpoint = get_initial_checkpoint();
  initialize(cg, input, state, checkpoint);

  std::vector<unsigned> actions;
  while (!state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    sys.get_valid_actions(state, valid_actions);

    dynet::expr::Expression score_exprs = get_scores(checkpoint);
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    auto payload = get_best_action(scores, valid_actions);
    unsigned best_a = payload.first;
    actions.push_back(best_a);
    perform_action(best_a, cg, state, checkpoint);
  }
  destropy_checkpoint(checkpoint);
  vector_to_parse(state.heads, state.deprels, parse);
}

void Parser::beam_search(dynet::ComputationGraph & cg,
                         const InputUnits & input,
                         const unsigned& beam_size,
                         bool structure_score,
                         std::vector<ParseUnits>& parses) {
  typedef std::tuple<unsigned, unsigned, float> Transition;

  new_graph(cg);
  unsigned len = input.size();
  std::vector<State> states;
  std::vector<float> scores;
  std::vector<StateCheckpoint*> checkpoints;

  states.push_back(State(len));
  scores.push_back(0.);
  StateCheckpoint * initial_checkpoint = get_initial_checkpoint();
  initialize(cg, input, states[0], initial_checkpoint);
  checkpoints.push_back(initial_checkpoint);

  unsigned curr = 0, next = 1;
  while (!states[curr].terminated()) {
    std::vector<Transition> transitions;
    for (unsigned i = curr; i < next; ++i) {
      const State& state = states[i];
      float score = scores[i];
      StateCheckpoint * checkpoint = checkpoints[i];

      std::vector<unsigned> valid_actions;
      sys.get_valid_actions(state, valid_actions);

      dynet::expr::Expression score_exprs = get_scores(checkpoint);
      if (!structure_score) { score_exprs = dynet::expr::log_softmax(score_exprs); }
      std::vector<float> s = dynet::as_vector(cg.get_value(score_exprs));
      for (unsigned a : valid_actions) {
        transitions.push_back(std::make_tuple(i, a, score + s[a]));
      }
    }

    sort(transitions.begin(), transitions.end(),
         [](const Transition& a, const Transition& b) { return std::get<2>(a) > std::get<2>(b); });
    curr = next;

    for (unsigned i = 0; i < transitions.size() && i < beam_size; ++i) {
      unsigned cursor = std::get<0>(transitions[i]);
      unsigned action = std::get<1>(transitions[i]);
      float new_score = std::get<2>(transitions[i]);


      State& state = states[cursor];
      State new_state(state);
      StateCheckpoint * new_checkpoint = copy_checkpoint(checkpoints[cursor]);
      perform_action(action, cg, new_state, new_checkpoint);

      //      
      states.push_back(new_state);
      scores.push_back(new_score);
      checkpoints.push_back(new_checkpoint);
      next++;
    }
  }
  for (StateCheckpoint * checkpoint : checkpoints) {
    destropy_checkpoint(checkpoint);
  }
  parses.resize(next - curr);
  for (unsigned i = curr; i < next; ++i) {
    vector_to_parse(states[i].heads, states[i].deprels, parses[i - curr]);
  }
}
*/