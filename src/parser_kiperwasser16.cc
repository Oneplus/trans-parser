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

std::vector<dynet::expr::Expression> Kiperwasser16ParserModel::get_params() {
  std::vector<dynet::expr::Expression> ret;
  for (auto & layer : fwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : bwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & e : merge_input.get_params()) { ret.push_back(e); }
  for (auto & e : merge.get_params()) { ret.push_back(e); }
  for (auto & e : scorer.get_params()) { ret.push_back(e); }
  ret.push_back(empty);
  ret.push_back(fwd_guard);
  ret.push_back(bwd_guard);
  return ret;
}

void Kiperwasser16ParserState::ArcEagerExtractor::extract(const TransitionState & state) {
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
  f0 = (stack_size > 3 ? encoded[state.stack[stack_size - 3]] : empty);
  f1 = (stack_size > 2 ? encoded[state.stack[stack_size - 2]] : empty);
  f2 = (stack_size > 1 ? encoded[state.stack[stack_size - 1]] : empty);

  unsigned buffer_size = state.buffer.size();
  f3 = (buffer_size > 1 ? encoded[state.buffer[buffer_size - 1]] : empty);
}

void Kiperwasser16ParserState::ArcHybridExtractor::extract(const TransitionState & state) {
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;

  unsigned stack_size = state.stack.size();
  f0 = (stack_size > 3 ? encoded[state.stack[stack_size - 3]] : empty);
  f1 = (stack_size > 2 ? encoded[state.stack[stack_size - 2]] : empty);
  f2 = (stack_size > 1 ? encoded[state.stack[stack_size - 1]] : empty);

  unsigned buffer_size = state.buffer.size();
  f3 = (buffer_size > 1 ? encoded[state.buffer[buffer_size - 1]] : empty);
}

void Kiperwasser16ParserState::SwapExtractor::extract(const TransitionState & state) {
  dynet::expr::Expression & empty = hook->model.empty;
  dynet::expr::Expression & f0 = hook->f0;
  dynet::expr::Expression & f1 = hook->f1;
  dynet::expr::Expression & f2 = hook->f2;
  dynet::expr::Expression & f3 = hook->f3;
  std::vector<dynet::expr::Expression> & encoded = hook->encoded;
  
  unsigned stack_size = state.stack.size();
  f0 = (stack_size > 3 ? encoded[state.stack[stack_size - 3]] : empty);
  f1 = (stack_size > 2 ? encoded[state.stack[stack_size - 2]] : empty);
  f2 = (stack_size > 1 ? encoded[state.stack[stack_size - 1]] : empty);

  unsigned buffer_size = state.buffer.size();
  f3 = (buffer_size > 1 ? encoded[state.buffer[buffer_size - 1]] : empty);
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
    _ERROR << "K16:: Unknown transition system: " << system_name;
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
  new_parser_state->encoded = encoded;
  return new_parser_state;
}

dynet::expr::Expression Kiperwasser16ParserState::get_scores() {
  return model.scorer.get_output(
    dynet::expr::tanh(model.merge.get_output(f0, f1, f2, f3)));
}

std::vector<dynet::expr::Expression> Kiperwasser16ParserState::get_params() {
  return model.get_params();
}

Kiperwasser16ParserStateBuilder::Kiperwasser16ParserStateBuilder(const po::variables_map & conf,
                                                                 dynet::Model & model,
                                                                 TransitionSystem & system,
                                                                 const Corpus & corpus,
                                                                 const Embeddings & pretrained) :
  ParserStateBuilder(model, system) {
  parser_model = new Kiperwasser16ParserModel(model,
                                              corpus.training_vocab.size() + 10,
                                              conf["word_dim"].as<unsigned>(),
                                              corpus.pos_map.size() + 10,
                                              conf["pos_dim"].as<unsigned>(),
                                              corpus.norm_map.size() + 1,
                                              conf["pretrained_dim"].as<unsigned>(),
                                              system.num_actions(),
                                              conf["layers"].as<unsigned>(),
                                              conf["lstm_input_dim"].as<unsigned>(),
                                              conf["hidden_dim"].as<unsigned>(),
                                              system,
                                              pretrained);
}

ParserState * Kiperwasser16ParserStateBuilder::build() {
  return new Kiperwasser16ParserState(*parser_model);
}
