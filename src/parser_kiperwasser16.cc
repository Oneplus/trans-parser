#include "parser_kiperwasser16.h"
#include "logging.h"

Kiperwasser16ParserModel::Kiperwasser16ParserModel(dynet::Model & m,
                                                   unsigned size_w,
                                                   unsigned dim_w,
                                                   unsigned size_p,
                                                   unsigned dim_p, 
                                                   unsigned size_t,
                                                   unsigned dim_t,
                                                   unsigned size_a,
                                                   unsigned n_layers,
                                                   unsigned dim_lstm_in,
                                                   unsigned dim_hidden,
                                                   TransitionSystem & system,
                                                   const Embeddings & pretrained) :
  ParserModel(system),
  fwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  bwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  preword_emb(m, size_t, dim_t, false),
  merge_input(m, dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
  scorer(m, dim_hidden, size_a),
  p_empty(m.add_parameters({ dim_hidden })),
  p_fwd_guard(m.add_parameters({ dim_lstm_in })),
  p_bwd_guard(m.add_parameters({ dim_lstm_in })),
  pretrained(pretrained),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  size_t(size_t), dim_t(dim_t),
  size_a(size_a),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {
  for (auto it : pretrained) {
    preword_emb.p_labels.initialize(it.first, it.second);
  }
}

void Kiperwasser16ParserModel::new_graph(dynet::ComputationGraph & cg) {
  fwd_lstm.new_graph(cg);
  bwd_lstm.new_graph(cg);
  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  preword_emb.new_graph(cg);
  merge_input.new_graph(cg);
  merge.new_graph(cg);
  scorer.new_graph(cg);

  fwd_guard = dynet::expr::parameter(cg, p_fwd_guard);
  bwd_guard = dynet::expr::parameter(cg, p_bwd_guard);
  empty = dynet::expr::parameter(cg, p_empty);
}

void Kiperwasser16ParserState::ArcEagerExtractor::extract(const TransitionState & state) {
  // S1, S0, B0, B1
  // should do after sys.perform_action
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;

  unsigned stack_size = state.stack.size();
  f0 = (stack_size > 2 ? encoded[state.stack[stack_size - 2]] : empty);
  f1 = (stack_size > 1 ? encoded[state.stack[stack_size - 1]] : empty);

  unsigned buffer_size = state.buffer.size();
  f2 = (buffer_size > 1 ? encoded[state.buffer[buffer_size - 1]] : empty);
  f3 = (buffer_size > 2 ? encoded[state.buffer[buffer_size - 2]] : empty);
}

void Kiperwasser16ParserState::ArcStandardExtractor::extract(const TransitionState & state) {
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;
  
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { f0 = encoded[state.stack[stack_size - 3]]; } else { f0 = empty; }
  if (stack_size > 2) { f1 = encoded[state.stack[stack_size - 2]]; } else { f1 = empty; }
  if (stack_size > 1) { f2 = encoded[state.stack[stack_size - 1]]; } else { f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { f3 = encoded[state.buffer[buffer_size - 1]]; } else { f3 = empty; }
}

void Kiperwasser16ParserState::ArcHybridExtractor::extract(const TransitionState & state) {
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;

  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { f0 = encoded[state.stack[stack_size - 3]]; } else { f0 = empty; }
  if (stack_size > 2) { f1 = encoded[state.stack[stack_size - 2]]; } else { f1 = empty; }
  if (stack_size > 1) { f2 = encoded[state.stack[stack_size - 1]]; } else { f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { f3 = encoded[state.buffer[buffer_size - 1]]; } else { f3 = empty; }
}

void Kiperwasser16ParserState::SwapExtractor::extract(const TransitionState & state) {
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;
  
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { f0 = encoded[state.stack[stack_size - 3]]; } else { f0 = empty; }
  if (stack_size > 2) { f1 = encoded[state.stack[stack_size - 2]]; } else { f1 = empty; }
  if (stack_size > 1) { f2 = encoded[state.stack[stack_size - 1]]; } else { f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { f3 = encoded[state.buffer[buffer_size - 1]]; } else { f3 = empty; }
}

Kiperwasser16ParserState::Kiperwasser16ParserState(Kiperwasser16ParserModel & model) : model(model) {
  std::string system_name = model.system.system_name();
  if (system_name == "arcstd") {
    extractor = new ArcStandardExtractor(this);
  } else if (system_name == "arceager") {
    extractor = new ArcEagerExtractor(this);
  } else if (system_name == "archybrid") {
    extractor = new ArcHybridExtractor(this);
  } else if (system_name == "swap") {
    extractor = new SwapExtractor(this);
  } else {
    _ERROR << "D15:: Unknown transition system: " << system_name;
    exit(1);
  }
}

void Kiperwasser16ParserState::new_graph(dynet::ComputationGraph & cg) {
  model.new_graph(cg);
}

void Kiperwasser16ParserState::initialize(dynet::ComputationGraph & cg,
                                          const InputUnits & input) {
  model.fwd_lstm.start_new_sequence();
  model.bwd_lstm.start_new_sequence();

  unsigned len = input.size();
  std::vector<dynet::expr::Expression> lstm_input(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;
    unsigned aux_wid = input[i].aux_wid;
    if (!model.pretrained.count(aux_wid)) { aux_wid = 0; }

    lstm_input[i] = dynet::expr::rectify(model.merge_input.get_output(
      model.word_emb.embed(wid),
      model.pos_emb.embed(pid), 
      model.preword_emb.embed(aux_wid))
    );
  }

  model.fwd_lstm.add_input(model.fwd_guard);
  model.bwd_lstm.add_input(model.bwd_guard);
  std::vector<dynet::expr::Expression> fwd_lstm_output(len);
  std::vector<dynet::expr::Expression> bwd_lstm_output(len);
  for (unsigned i = 0; i < len; ++i) {
    model.fwd_lstm.add_input(lstm_input[i]);
    model.bwd_lstm.add_input(lstm_input[len - 1 - i]);
    fwd_lstm_output[i] = model.fwd_lstm.back();
    bwd_lstm_output[len - 1 - i] = model.bwd_lstm.back();
  }
  encoded.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    encoded[i] = dynet::expr::concatenate({ fwd_lstm_output[i], bwd_lstm_output[i] });
  }

  TransitionState state(len);
  state.initialize(input);
  extractor->extract(state);
}

void Kiperwasser16ParserState::perform_action(const unsigned & action,
                                              dynet::ComputationGraph & cg,
                                              const TransitionState & state) {
  extractor->extract(state);
}

ParserState * Kiperwasser16ParserState::copy() {
  Kiperwasser16ParserState * new_parser_state = new Kiperwasser16ParserState(model);
  new_parser_state->f0 = f0;
  new_parser_state->f1 = f1;
  new_parser_state->f2 = f2;
  new_parser_state->f3 = f3;
  return new_parser_state;
}

dynet::expr::Expression Kiperwasser16ParserState::get_scores() {
  return dynet::expr::tanh(model.merge.get_output(f0, f1, f2, f3));
}
