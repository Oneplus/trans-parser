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
