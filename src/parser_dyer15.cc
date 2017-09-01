#include "parser_dyer15.h"
#include "dynet/expr.h"
#include "corpus.h"
#include "logging.h"
#include "arceager.h"
#include "arcstd.h"
#include "archybrid.h"
#include "swap.h"
#include <vector>
#include <random>

Dyer15ParserModel::Dyer15ParserModel(dynet::Model& m,
                                     unsigned size_w,  //
                                     unsigned dim_w,   // word size, word dim
                                     unsigned size_p,  //
                                     unsigned dim_p,   // pos size, pos dim
                                     unsigned size_t,  //
                                     unsigned dim_t,   // pword size, pword dim
                                     unsigned size_l,
                                     unsigned dim_l,
                                     unsigned size_a,  //
                                     unsigned dim_a,   // act size, act dim
                                     unsigned n_layers,
                                     unsigned dim_lstm_in,
                                     unsigned dim_hidden,
                                     TransitionSystem & system,
                                     const Embeddings& pretrained) :
  ParserModel(system),
  s_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  q_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  a_lstm(n_layers, dim_a, dim_hidden, m),
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  preword_emb(m, size_t, dim_t, false),
  act_emb(m, size_a, dim_a),
  rel_emb(m, size_l + 1, dim_l),
  merge_input(m, dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
  composer(m, dim_lstm_in, dim_lstm_in, dim_l, dim_lstm_in),
  scorer(m, dim_hidden, size_a),
  p_action_start(m.add_parameters({ dim_a })),
  p_buffer_guard(m.add_parameters({ dim_lstm_in })),
  p_stack_guard(m.add_parameters({ dim_lstm_in })),
  pretrained(pretrained),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  size_t(size_t), dim_t(dim_t),
  size_l(size_l), dim_l(dim_l), size_a(size_a), dim_a(dim_a),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {
  for (auto it : pretrained) {
    preword_emb.p_e.initialize(it.first, it.second);
  }
}

void Dyer15ParserModel::new_graph(dynet::ComputationGraph & cg) {
  s_lstm.new_graph(cg);
  q_lstm.new_graph(cg);
  a_lstm.new_graph(cg);

  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  preword_emb.new_graph(cg);
  act_emb.new_graph(cg);
  rel_emb.new_graph(cg);

  merge_input.new_graph(cg);
  merge.new_graph(cg);
  composer.new_graph(cg);
  scorer.new_graph(cg);

  action_start = dynet::parameter(cg, p_action_start);
  buffer_guard = dynet::parameter(cg, p_buffer_guard);
  stack_guard = dynet::parameter(cg, p_stack_guard);
}

std::vector<dynet::Expression> Dyer15ParserModel::get_params() {
  std::vector<dynet::Expression> ret;
  for (auto & layer : s_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : q_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : a_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & e : merge_input.get_params()) { ret.push_back(e); }
  for (auto & e : merge.get_params()) { ret.push_back(e); }
  for (auto & e : composer.get_params()) { ret.push_back(e); }
  for (auto & e : scorer.get_params()) { ret.push_back(e); }
  ret.push_back(buffer_guard);
  ret.push_back(stack_guard);
  ret.push_back(action_start);
  return ret;
}

void Dyer15ParserState::ArcEagerPerformer::perform_action(const unsigned & action,
                                                          dynet::ComputationGraph & cg) {
  dynet::Expression act_expr = state->model.act_emb.embed(action);
  unsigned _, deprel; state->model.system.split(action, _, deprel);
  dynet::Expression rel_expr = state->model.rel_emb.embed((deprel == UINT_MAX ? state->model.size_l: deprel));
  dynet::CoupledLSTMBuilder & a_lstm = state->model.a_lstm;
  dynet::CoupledLSTMBuilder & s_lstm = state->model.s_lstm;
  dynet::CoupledLSTMBuilder & q_lstm = state->model.q_lstm;
  dynet::RNNPointer & a_pointer = state->a_pointer;
  dynet::RNNPointer & s_pointer = state->s_pointer;
  dynet::RNNPointer & q_pointer = state->q_pointer;
  Merge3Layer & composer = state->model.composer;
  std::vector<dynet::Expression> & stack = state->stack;
  std::vector<dynet::Expression> & buffer = state->buffer;

  a_lstm.add_input(a_pointer, act_expr);
  a_pointer = a_lstm.state();

  if (ArcEager::is_shift(action)) {
    const dynet::Expression & buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcEager::is_left(action)) {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = buffer.back();
    mod_expr = stack.back();

    stack.pop_back();
    buffer.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    q_pointer = q_lstm.get_head(q_pointer);
    buffer.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else if (ArcEager::is_right(action)) {
    dynet::Expression mod_expr, hed_expr;
    mod_expr = buffer.back();
    hed_expr = stack.back();

    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    stack.push_back(mod_expr);
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else {
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
  }
}

void Dyer15ParserState::ArcStandardPerformer::perform_action(const unsigned & action,
                                                             dynet::ComputationGraph & cg) {
  dynet::Expression act_expr = state->model.act_emb.embed(action);
  unsigned _, deprel; state->model.system.split(action, _, deprel);
  dynet::Expression rel_expr = state->model.rel_emb.embed((deprel == UINT_MAX ? state->model.size_l: deprel));
  dynet::CoupledLSTMBuilder & a_lstm = state->model.a_lstm;
  dynet::CoupledLSTMBuilder & s_lstm = state->model.s_lstm;
  dynet::CoupledLSTMBuilder & q_lstm = state->model.q_lstm;
  dynet::RNNPointer & a_pointer = state->a_pointer;
  dynet::RNNPointer & s_pointer = state->s_pointer;
  dynet::RNNPointer & q_pointer = state->q_pointer;
  Merge3Layer & composer = state->model.composer;
  std::vector<dynet::Expression> & stack = state->stack;
  std::vector<dynet::Expression> & buffer = state->buffer;

  a_lstm.add_input(a_pointer, act_expr);
  a_pointer = a_lstm.state();
  if (ArcStandard::is_shift(action)) {
    const dynet::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else {
    dynet::Expression mod_expr, hed_expr;
    if (ArcStandard::is_left(action)) {
      hed_expr = stack.back();
      mod_expr = stack[stack.size() - 2];
    } else {
      mod_expr = stack.back();
      hed_expr = stack[stack.size() - 2];
    }
    stack.pop_back(); stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);

    stack.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

void Dyer15ParserState::ArcHybridPerformer::perform_action(const unsigned & action,
                                                           dynet::ComputationGraph & cg) {
  dynet::Expression act_expr = state->model.act_emb.embed(action);
  unsigned _, deprel; state->model.system.split(action, _, deprel);
  dynet::Expression rel_expr = state->model.rel_emb.embed((deprel == UINT_MAX ? state->model.size_l: deprel));
  dynet::CoupledLSTMBuilder & a_lstm = state->model.a_lstm;
  dynet::CoupledLSTMBuilder & s_lstm = state->model.s_lstm;
  dynet::CoupledLSTMBuilder & q_lstm = state->model.q_lstm;
  dynet::RNNPointer & a_pointer = state->a_pointer;
  dynet::RNNPointer & s_pointer = state->s_pointer;
  dynet::RNNPointer & q_pointer = state->q_pointer;
  Merge3Layer & composer = state->model.composer;
  std::vector<dynet::Expression> & stack = state->stack;
  std::vector<dynet::Expression> & buffer = state->buffer;

  a_lstm.add_input(a_pointer, act_expr);
  a_pointer = a_lstm.state();
  if (ArcHybrid::is_shift(action)) {
    const dynet::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcHybrid::is_left(action)) {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = buffer.back();
    mod_expr = stack.back();

    stack.pop_back();
    buffer.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    q_pointer = q_lstm.get_head(q_pointer);
    buffer.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = stack[stack.size() - 2];
    mod_expr = stack.back();

    stack.pop_back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

void Dyer15ParserState::SwapPerformer::perform_action(const unsigned & action,
                                                      dynet::ComputationGraph & cg) {
  dynet::Expression act_expr = state->model.act_emb.embed(action);
  unsigned _, deprel; state->model.system.split(action, _, deprel);
  dynet::Expression rel_expr = state->model.rel_emb.embed((deprel == UINT_MAX ? state->model.size_l: deprel));
  dynet::CoupledLSTMBuilder & a_lstm = state->model.a_lstm;
  dynet::CoupledLSTMBuilder & s_lstm = state->model.s_lstm;
  dynet::CoupledLSTMBuilder & q_lstm = state->model.q_lstm;
  dynet::RNNPointer & a_pointer = state->a_pointer;
  dynet::RNNPointer & s_pointer = state->s_pointer;
  dynet::RNNPointer & q_pointer = state->q_pointer;
  Merge3Layer & composer = state->model.composer;
  std::vector<dynet::Expression> & stack = state->stack;
  std::vector<dynet::Expression> & buffer = state->buffer;
  
  a_lstm.add_input(a_pointer, act_expr);
  a_pointer = a_lstm.state();
  if (Swap::is_shift(action)) {
    const dynet::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (Swap::is_swap(action)) {
    dynet::Expression j_expr = stack.back();
    dynet::Expression i_expr = stack[stack.size() - 2];

    stack.pop_back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(j_expr);
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    buffer.push_back(i_expr);
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else {
    dynet::Expression mod_expr, hed_expr;
    if (Swap::is_left(action)) {
      hed_expr = stack.back();
      mod_expr = stack[stack.size() - 2];
    } else {
      hed_expr = stack[stack.size() - 2];
      mod_expr = stack.back();
    }
    stack.pop_back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

Dyer15ParserState::Dyer15ParserState(Dyer15ParserModel & model) : model(model) {
  std::string system_name = model.system.system_name();
  if (system_name == "arcstd") {
    performer = new ArcStandardPerformer(this);
  } else if (system_name == "arceager") {
    performer = new ArcEagerPerformer(this);
  } else if (system_name == "archybrid") {
    performer = new ArcHybridPerformer(this);
  } else if (system_name == "swap") {
    performer = new SwapPerformer(this);
  } else {
    _ERROR << "D15:: Unknown transition system: " << system_name;
    exit(1);
  }
}

void Dyer15ParserState::new_graph(dynet::ComputationGraph & cg) {
  model.new_graph(cg);
}

void Dyer15ParserState::initialize(dynet::ComputationGraph & cg,
                                   const InputUnits & input) {
  model.s_lstm.start_new_sequence();
  model.q_lstm.start_new_sequence();
  model.a_lstm.start_new_sequence();
  model.a_lstm.add_input(model.action_start);

  unsigned len = input.size();
  stack.clear();
  buffer.resize(len + 1);

  // Pay attention to this, if the guard word is handled here, there is no need
  // to insert it when loading the data.
  buffer[0] = model.buffer_guard;
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;
    unsigned nid = input[i].nid;
    if (!model.pretrained.count(nid)) { nid = 0; }

    buffer[len - i] = dynet::rectify(model.merge_input.get_output(
      model.word_emb.embed(wid),
      model.pos_emb.embed(pid),
      model.preword_emb.embed(nid)
    ));
  }

  // push word into buffer in reverse order, pay attention to (i == len).
  for (unsigned i = 0; i <= len; ++i) {
    model.q_lstm.add_input(buffer[i]);
  }

  model.s_lstm.add_input(model.stack_guard);
  stack.push_back(model.stack_guard);
  a_pointer = model.a_lstm.state();
  s_pointer = model.s_lstm.state();
  q_pointer = model.q_lstm.state();
}

void Dyer15ParserState::perform_action(const unsigned & action,
                                       dynet::ComputationGraph & cg,
                                       const TransitionState & state) {
  performer->perform_action(action, cg);
}

ParserState * Dyer15ParserState::copy() {
  Dyer15ParserState * new_parser_state = new Dyer15ParserState(model);
  new_parser_state->s_pointer = s_pointer;
  new_parser_state->q_pointer = q_pointer;
  new_parser_state->a_pointer = a_pointer;
  new_parser_state->stack = stack;
  new_parser_state->buffer = buffer;
  return new_parser_state;
}

dynet::Expression Dyer15ParserState::get_scores() {
  return model.scorer.get_output(dynet::rectify(model.merge.get_output(
    model.s_lstm.get_h(s_pointer).back(),
    model.q_lstm.get_h(q_pointer).back(),
    model.a_lstm.get_h(a_pointer).back())
  ));
}

std::vector<dynet::Expression> Dyer15ParserState::get_params() {
  return model.get_params();
}

Dyer15ParserStateBuilder::Dyer15ParserStateBuilder(const po::variables_map & conf,
                                                   dynet::Model & model,
                                                   TransitionSystem & system,
                                                   const Corpus & corpus,
                                                   const Embeddings & pretrained) :
  ParserStateBuilder(model, system) {
  parser_model = new Dyer15ParserModel(model,
                                       corpus.training_vocab.size() + 10,
                                       conf["word_dim"].as<unsigned>(),
                                       corpus.pos_map.size() + 10,
                                       conf["pos_dim"].as<unsigned>(),
                                       corpus.norm_map.size() + 1,
                                       conf["pretrained_dim"].as<unsigned>(),
                                       system.num_deprels(),
                                       conf["label_dim"].as<unsigned>(),
                                       system.num_actions(),
                                       conf["action_dim"].as<unsigned>(),
                                       conf["layers"].as<unsigned>(),
                                       conf["lstm_input_dim"].as<unsigned>(),
                                       conf["hidden_dim"].as<unsigned>(),
                                       system,
                                       pretrained);
}

ParserState * Dyer15ParserStateBuilder::build() {
  return new Dyer15ParserState(*parser_model);
}
