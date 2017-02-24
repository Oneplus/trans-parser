#include "system.h"
#include "corpus.h"


TransitionState::TransitionState(unsigned n) :
  heads(n, Corpus::BAD_HED),
  deprels(n, Corpus::BAD_DEL) {
}

void TransitionState::initialize(const InputUnits & input) {
  unsigned len = input.size();
  buffer.resize(len + 1);
  for (unsigned i = 0; i < len; ++i) { buffer[len - i] = i; }
  buffer[0] = Corpus::BAD_HED;
  stack.push_back(Corpus::BAD_HED);
}

bool TransitionState::terminated() const {
  return !(stack.size() > 2 || buffer.size() > 1);
}
