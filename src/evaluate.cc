#include "evaluate.h"
#include "logging.h"
#include "sys_utils.h"
#include <fstream>
#include <chrono>

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               ParserStateBuilder & state_builder,
               const std::string & output) {
  TransitionSystem & system = state_builder.system;
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  std::ofstream ofs(output);

  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse = corpus.devel_parses[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.training_vocab.count(u.wid)) { u.wid = kUNK; }
    }
    ParseUnits result;
    ParserState * parser_state = state_builder.build();
    dynet::ComputationGraph cg;
    parser_state->new_graph(cg);
    parser_state->initialize(cg, input_units);

    unsigned len = input_units.size();
    TransitionState transition_state(len);
    transition_state.initialize(input_units);

    while (!transition_state.terminated()) {
      std::vector<unsigned> valid_actions;
      system.get_valid_actions(transition_state, valid_actions);

      dynet::Expression score_exprs = parser_state->get_scores();
      std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));
    
      auto payload = ParserState::get_best_action(scores, valid_actions);
      unsigned best_a = payload.first;
      system.perform_action(transition_state, best_a);
      parser_state->perform_action(best_a, cg, transition_state);
    }
    delete parser_state;

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }
    vector_to_parse(transition_state.heads, transition_state.deprels, result);

    // pay attention to this, not counting the last DUMMY_ROOT
    for (unsigned i = 0; i < len - 1; ++i) {
      ofs << i + 1 << "\t"                //  id
        << input_units[i].w_str << "\t"   //  form
        << input_units[i].n_str << "\t"   //  lemma
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << input_units[i].f_str << "\t"
        << (parse[i].head >= len ? 0 : parse[i].head) << "\t"
        << corpus.deprel_map.get(parse[i].deprel) << "\t"
        << (result[i].head >= len ? 0 : result[i].head) << "\t"
        << corpus.deprel_map.get(result[i].deprel)
        << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>(), output);
  _INFO << "Evaluate:: UAS " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               std::vector<ParserStateBuilder*>& pretrained_state_builders,
               const std::string & output) {
  unsigned n_pretrained = pretrained_state_builders.size();
  assert(n_pretrained > 0);
  TransitionSystem & system = pretrained_state_builders[0]->system;
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  std::ofstream ofs(output);

  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse = corpus.devel_parses[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.training_vocab.count(u.wid)) { u.wid = kUNK; }
    }
    ParseUnits result;
    std::vector<ParserState *> parser_states(n_pretrained);
    dynet::ComputationGraph cg;
    for (unsigned i = 0; i < n_pretrained; ++i) {
      parser_states[i] = pretrained_state_builders[i]->build();
      parser_states[i]->new_graph(cg);
      parser_states[i]->initialize(cg, input_units);
    }

    unsigned len = input_units.size();
    TransitionState transition_state(len);
    transition_state.initialize(input_units);

    while (!transition_state.terminated()) {
      std::vector<unsigned> valid_actions;
      system.get_valid_actions(transition_state, valid_actions);

      std::vector<float> scores;
      for (ParserState* parser_state : parser_states) {
        dynet::Expression score_exprs = parser_state->get_scores();
        std::vector<float> score = dynet::as_vector(cg.get_value(score_exprs));
        if (scores.size() == 0) {
          scores = score;
        } else {
          for (unsigned i = 0; i < score.size(); ++i) {
            scores[i] += score[i];
          }
        }
      }
      auto payload = ParserState::get_best_action(scores, valid_actions);
      unsigned best_a = payload.first;
      system.perform_action(transition_state, best_a);
      for (ParserState * parser_state : parser_states) {
        parser_state->perform_action(best_a, cg, transition_state);
      }
    }
    for (ParserState * parser_state : parser_states) {
      delete parser_state;
    }

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }
    vector_to_parse(transition_state.heads, transition_state.deprels, result);

    // pay attention to this, not counting the last DUMMY_ROOT
    for (unsigned i = 0; i < len - 1; ++i) {
      ofs << i + 1 << "\t"                //  id
        << input_units[i].w_str << "\t"   //  form
        << input_units[i].n_str << "\t"   //  lemma
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << input_units[i].f_str << "\t"
        << (parse[i].head >= len ? 0 : parse[i].head) << "\t"
        << corpus.deprel_map.get(parse[i].deprel) << "\t"
        << (result[i].head >= len ? 0 : result[i].head) << "\t"
        << corpus.deprel_map.get(result[i].deprel)
        << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>(), output);
  _INFO << "Evaluate:: UAS " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

float beam_search(const po::variables_map & conf,
                  Corpus & corpus,
                  ParserStateBuilder & state_builder,
                  const std::string & output,
                  bool structure) {
  typedef std::tuple<unsigned, unsigned, float> Transition;
  TransitionSystem & system = state_builder.system;

  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  unsigned beam_size = conf["beam_size"].as<unsigned>();

  std::ofstream ofs(output);
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse = corpus.devel_parses[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.training_vocab.count(u.wid)) { u.wid = kUNK; }
    }

    std::vector<TransitionState> transition_states;
    std::vector<float> scores;
    std::vector<ParserState *> parser_states;

    parser_states.push_back(state_builder.build());
    dynet::ComputationGraph cg;
    parser_states[0]->new_graph(cg);
    parser_states[0]->initialize(cg, input_units);

    unsigned len = input_units.size();
    transition_states.push_back(TransitionState(len));
    transition_states[0].initialize(input_units);
    
    scores.push_back(0.);

    unsigned curr = 0, next = 1;
    while (!transition_states[curr].terminated()) {
      std::vector<Transition> transitions;
      for (unsigned i = curr; i < next; ++i) {
        const TransitionState & transition_state = transition_states[i];
        float score = scores[i];

        ParserState * parser_state = parser_states[i];

        std::vector<unsigned> valid_actions;
        system.get_valid_actions(transition_state, valid_actions);
      
        dynet::Expression score_exprs = parser_state->get_scores();
        if (!structure) { score_exprs = dynet::log_softmax(score_exprs); }
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

        TransitionState & transition_state = transition_states[cursor];
        TransitionState new_transition_state(transition_state);
        ParserState * parser_state = parser_states[cursor];
        ParserState * new_parser_state = parser_state->copy();

        if (action != system.num_actions()) {
          system.perform_action(new_transition_state, action);
          new_parser_state->perform_action(action, cg, new_transition_state);
        }

        transition_states.push_back(new_transition_state);
        scores.push_back(new_score);
        parser_states.push_back(new_parser_state);
        next++;
      }
    }

    for (ParserState * parser_state : parser_states) {
      delete parser_state;
    }

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }
    
    ParseUnits result; 
    vector_to_parse(transition_states[curr].heads, transition_states[curr].deprels, result);

    for (unsigned i = 0; i < len - 1; ++i) {
      ofs << i + 1 << "\t"                //  id
        << input_units[i].w_str << "\t"   //  form
        << input_units[i].n_str << "\t"   //  lemma
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << input_units[i].f_str << "\t"
        << (parse[i].head >= len ? 0 : parse[i].head) << "\t"
        << corpus.deprel_map.get(parse[i].deprel) << "\t"
        << (result[i].head >= len ? 0 : result[i].head) << "\t"
        << corpus.deprel_map.get(result[i].deprel)
        << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>(), output);
  _INFO << "Evaluate:: UAS " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

void generate_static_ensemble_data(const po::variables_map & conf,
                                   Corpus & corpus,
                                   std::vector<ParserStateBuilder*>& pretrained_state_builders, 
                                   const std::string & output) {
}
