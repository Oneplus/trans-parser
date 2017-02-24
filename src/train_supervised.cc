#include "trainer_utils.h"
#include "train_supervised.h"
#include "tree.h"
#include "logging.h"
#include "evaluate.h"

po::options_description SupervisedTrainer::get_options() {
  po::options_description cmd("Supervised options");
  cmd.add_options()
    ("supervised_oracle", po::value<std::string>()->default_value("static"), "The type of oracle in supervised learning [static|dynamic].")
    ("supervised_objective", po::value<std::string>()->default_value("crossentropy"), "The learning objective [crossentropy|rank|bipartie_rank|structure]")
    ("supervised_do_pretrain_iter", po::value<unsigned>()->default_value(1), "The number of pretrain iteration on dynamic oracle.")
    ("supervised_do_explore_prob", po::value<float>()->default_value(0.9), "The probability of exploration.")
    ;
  return cmd;
}

SupervisedTrainer::SupervisedTrainer(const po::variables_map& conf,
                                     const Noisifier& noisifier,
                                     ParserStateBuilder & state_builder) :
  state_builder(state_builder),
  noisifier(noisifier) {
  if (conf["supervised_oracle"].as<std::string>() == "static") {
    oracle_type = kStatic;
  } else if (conf["supervised_oracle"].as<std::string>() == "dynamic") {
    oracle_type = kDynamic;
  } else {
    _ERROR << "Unknown oracle :" << conf["supervised_oracle"].as<std::string>();
  }

  std::string supervised_objective_name = 
    conf["supervised_objective"].as<std::string>();
  if (supervised_objective_name == "crossentropy") {
    objective_type = kCrossEntropy;
  } else if (supervised_objective_name == "rank") {
    objective_type = kRank;
  } else if (supervised_objective_name == "bipartie_rank") {
    objective_type = kBipartieRank;
  } else {
    objective_type = kStructure;
    if (!conf.count("beam_size") || conf["beam_size"].as<unsigned>() <= 1) {
      _ERROR << "SUP:: set structure learning objective, but beam size unset.";
      exit(1);
    }
  }                                                                       
  _INFO << "SUP:: learning objective " << conf["supervised_objective"].as<std::string>();
  if (oracle_type == kDynamic) {
    do_pretrain_iter = conf["supervised_do_pretrain_iter"].as<unsigned>();
    do_explore_prob = conf["supervised_do_explore_prob"].as<float>();
    _INFO << "SUP:: use dynamic oracle training";
    _INFO << "SUP:: pretrain iteration = " << do_pretrain_iter;
    _INFO << "SUP:: explore prob = " << do_explore_prob;
  }
}

void SupervisedTrainer::train(const po::variables_map& conf,
                              Corpus& corpus,
                              const std::string& name,
                              const std::string& output,
                              bool allow_nonprojective,
                              bool allow_partial_tree) {
  dynet::Model& model = state_builder.model;
  _INFO << "SUP:: start lstm-parser supervised training.";

  dynet::Trainer* trainer = get_trainer(conf, model);
  unsigned max_iter = conf["max_iter"].as<unsigned>();

  float llh = 0.f, llh_in_batch = 0.f, best_f = 0.f;
  std::vector<unsigned> order;
  get_orders(corpus, order, allow_nonprojective, allow_partial_tree);
  float n_train = order.size();

  unsigned logc = 0;
  unsigned report_stops = conf["report_stops"].as<unsigned>();
  unsigned evaluate_stops = conf["evaluate_stops"].as<unsigned>();
  unsigned evaluate_skips = conf["evaluate_skips"].as<unsigned>();
  unsigned beam_size = (conf.count("beam_size") ? conf["beam_size"].as<unsigned>() : 0);
  bool use_beam_search = (beam_size > 1);
  _INFO << "SUP:: will stop after " << max_iter << " iterations.";
  for (unsigned iter = 0; iter < max_iter; ++iter) {
    llh = 0;
    _INFO << "SUP:: start training iteration #" << iter << ", shuffled.";
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits& input_units = corpus.training_inputs[sid];
      const ParseUnits& parse_units = corpus.training_parses[sid];
      
      noisifier.noisify(input_units);
      float lp;
      if (!allow_partial_tree) {
        if (objective_type == kStructure) {
          lp = train_structure_full_tree(input_units, parse_units, trainer, beam_size, iter);
        } else {
          lp = train_full_tree(input_units, parse_units, trainer, iter);
        }
      } else {
        lp = train_partial_tree(input_units, parse_units, trainer, iter);
      }

      llh += lp;
      llh_in_batch += lp;
      noisifier.denoisify(input_units);

      ++logc; 
      if (logc % report_stops == 0) {
        float epoch = (float(logc) / n_train);
        _INFO << "SUP:: iter #" << iter << " (epoch " << epoch << ") loss " << llh_in_batch;
        llh_in_batch = 0.f;
      }
      if (iter >= evaluate_skips && logc % evaluate_stops == 0) {
        float f = (use_beam_search ? beam_search(conf, corpus, state_builder, output) : evaluate(conf, corpus, state_builder, output));
        if (f > best_f) {
          best_f = f;
          _INFO << "SUP:: new best record achieved: " << best_f << ", saved.";
          dynet::save_dynet_model(name, (&model));
        }
      }
    }

    _INFO << "SUP:: end of iter #" << iter << " loss " << llh;
    float f = (use_beam_search ? beam_search(conf, corpus, state_builder, output) : evaluate(conf, corpus, state_builder, output));
    if (f > best_f) {
      best_f = f;
      _INFO << "SUP:: new best record achieved: " << best_f << ", saved.";
      dynet::save_dynet_model(name, (&model));
    }
    trainer->update_epoch();
    trainer->status();
  }

  delete trainer;
}

void SupervisedTrainer::add_loss_one_step(dynet::expr::Expression & score_expr,
                                          const unsigned & best_gold_action,
                                          const unsigned & worst_gold_action,
                                          const unsigned & best_non_gold_action,
                                          std::vector<dynet::expr::Expression> & loss) {
  TransitionSystem & system = state_builder.parser_model->get_system();
  unsigned illegal_action = system.num_actions();

  if (objective_type == kCrossEntropy) {
    loss.push_back(dynet::expr::pickneglogsoftmax(score_expr, best_gold_action));
  } else if (objective_type == kRank) {
    if (best_gold_action != illegal_action && best_non_gold_action != illegal_action) {
      loss.push_back(dynet::expr::pairwise_rank_loss(
        dynet::expr::pick(score_expr, best_gold_action),
        dynet::expr::pick(score_expr, best_non_gold_action)
      ));
    }
  } else {
    if (worst_gold_action != illegal_action && best_non_gold_action != illegal_action) {
      loss.push_back(dynet::expr::pairwise_rank_loss(
        dynet::expr::pick(score_expr, worst_gold_action),
        dynet::expr::pick(score_expr, best_non_gold_action)
      ));
    }
  }
}

float SupervisedTrainer::train_full_tree(const InputUnits& input_units,
                                         const ParseUnits& parse_units,
                                         dynet::Trainer* trainer,
                                         unsigned iter) {
  TransitionSystem & system = state_builder.parser_model->get_system();

  std::vector<unsigned> ref_heads, ref_deprels;
  parse_to_vector(parse_units, ref_heads, ref_deprels);
  std::vector<unsigned> gold_actions;
  system.get_oracle_actions(ref_heads, ref_deprels, gold_actions);

  ParserState * parser_state = state_builder.build();
  dynet::ComputationGraph cg;
  parser_state->new_graph(cg);
  parser_state->initialize(cg, input_units);

  unsigned len = input_units.size();
  TransitionState transition_state(len);
  transition_state.initialize(input_units);

  unsigned illegal_action = system.num_actions();
  unsigned n_actions = 0;

  std::vector<dynet::expr::Expression> loss;
  while (!transition_state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(transition_state, valid_actions);

    dynet::expr::Expression score_exprs = parser_state->get_scores();
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    unsigned action = 0;
    unsigned best_gold_action = illegal_action;
    unsigned worst_gold_action = illegal_action;
    unsigned best_non_gold_action = illegal_action;

    if (oracle_type == kDynamic) {
      auto payload = ParserState::get_best_action(scores, valid_actions);
      action = payload.first;
      std::vector<float> costs; // the larger, the better
      system.get_transition_costs(transition_state, valid_actions, ref_heads, ref_deprels, costs);
      float gold_action_cost = (*std::max_element(costs.begin(), costs.end()));
      float action_cost = 0.f;
      float best_gold_action_score = -1e10, worst_gold_action_score = 1e10, best_non_gold_action_score = -1e10;
      for (unsigned i = 0; i < valid_actions.size(); ++i) {
        unsigned act = valid_actions[i];
        float s = scores[act];
        if (costs[i] == gold_action_cost) {
          if (best_gold_action_score < s) { best_gold_action_score = s; best_gold_action = act; }
          if (worst_gold_action_score > s) { worst_gold_action_score = s; worst_gold_action = act; }
        } else {
          if (best_non_gold_action_score < s) { best_non_gold_action_score = s; best_non_gold_action = act; }
        }
        if (act == action) { action_cost = costs[i]; }
      }
      if (gold_action_cost != action_cost) {
        if (!(iter >= do_pretrain_iter && dynet::rand01() < do_explore_prob)) {
          action = best_gold_action;
        }
      }
    } else {
      best_gold_action = gold_actions[n_actions];
      action = gold_actions[n_actions];
      if (objective_type == kRank || objective_type == kBipartieRank) {
        float best_non_gold_action_score = -1e10;
        for (unsigned i = 0; i < valid_actions.size(); ++i) {
          unsigned act = valid_actions[i];
          if (act != best_gold_action && (scores[act] > best_non_gold_action_score)) {
            best_non_gold_action = act;
            best_non_gold_action_score = scores[act];
          }
        }
      }
    }
    
    add_loss_one_step(score_exprs, best_gold_action, worst_gold_action, best_non_gold_action, loss);

    system.perform_action(transition_state, action);
    parser_state->perform_action(action, cg, transition_state);
    n_actions++;
  }
  float ret = 0.;
  if (loss.size() > 0) {
    dynet::expr::Expression l = dynet::expr::sum(loss);
    ret = dynet::as_scalar(cg.forward(l));
    cg.backward(l);
    trainer->update(1.f);
  }
  delete parser_state;
  return ret;
}

float SupervisedTrainer::train_structure_full_tree(const InputUnits & input_units,
                                                   const ParseUnits & parse_units,
                                                   dynet::Trainer * trainer,
                                                   unsigned beam_size,
                                                   unsigned iter) {
  typedef std::tuple<unsigned, unsigned, float, dynet::expr::Expression> Transition;
  TransitionSystem & system = state_builder.parser_model->get_system();

  std::vector<unsigned> ref_heads, ref_deprels, gold_actions;
  parse_to_vector(parse_units, ref_heads, ref_deprels);
  system.get_oracle_actions(ref_heads, ref_deprels, gold_actions);

  unsigned len = input_units.size();
  std::vector<TransitionState> transition_states;
  std::vector<float> scores;
  std::vector<Expression> scores_exprs;
  std::vector<ParserState *> parser_states;

  dynet::ComputationGraph cg;

  transition_states.push_back(TransitionState(len));
  transition_states[0].initialize(input_units);

  parser_states.push_back(state_builder.build());
  parser_states[0]->initialize(cg, input_units);

  scores.push_back(0.);
  scores_exprs.push_back(dynet::expr::zeroes(cg, { 1 }));

  unsigned curr = 0, next = 1, corr = 0;
  unsigned n_step = 0;
  while (!transition_states[corr].terminated()) {
    unsigned gold_action = gold_actions[n_step];
    n_step++;

    std::vector<Transition> transitions;
    for (unsigned i = curr; i < next; ++i) {
      const TransitionState & prev_state = transition_states[i];
      float prev_score = scores[i];
      dynet::expr::Expression prev_score_expr = scores_exprs[i];

      if (prev_state.terminated()) {
        transitions.push_back(std::make_tuple(i, system.num_actions(), prev_score, prev_score_expr));
      } else {
        ParserState * parser_state = parser_states[i];
        std::vector<unsigned> valid_actions;
        system.get_valid_actions(prev_state, valid_actions);

        dynet::expr::Expression transit_scores_expr = parser_state->get_scores();
        std::vector<float> transit_scores = dynet::as_vector(cg.get_value(transit_scores_expr));
        for (unsigned a : valid_actions) {
          transitions.push_back(std::make_tuple(
            i, a, prev_score + transit_scores[a],
            prev_score_expr + dynet::expr::pick(transit_scores_expr, a)
          ));
        }
      }
    }

    sort(transitions.begin(), transitions.end(),
         [](const Transition& a, const Transition& b) { return std::get<2>(a) > std::get<2>(b); });

    unsigned new_corr = UINT_MAX, new_curr = next, new_next = next;
    for (unsigned i = 0; i < transitions.size() && i < beam_size; ++i) {
      unsigned cursor = std::get<0>(transitions[i]);
      unsigned action = std::get<1>(transitions[i]);
      float new_score = std::get<2>(transitions[i]);
      dynet::expr::Expression new_score_expr = std::get<3>(transitions[i]);
      TransitionState & transition_state = transition_states[cursor];

      TransitionState new_transition_state(transition_state);
      ParserState * parser_state = parser_states[i];
      ParserState * new_parser_state = parser_state->copy();
      if (action != system.num_actions()) {
        system.perform_action(new_transition_state, action);
        new_parser_state->perform_action(action, cg, new_transition_state);
      }

      //      
      transition_states.push_back(new_transition_state);
      scores.push_back(new_score);
      scores_exprs.push_back(new_score_expr);
      parser_states.push_back(new_parser_state);

      if (cursor == corr && action == gold_action) { new_corr = new_next; }
      new_next++;
    }
    if (new_corr == UINT_MAX) {
      // early stopping
      break;
    } else {
      corr = new_corr;
      curr = new_curr;
      next = new_next;
    }
  }

  for (ParserState * parser_state : parser_states) { delete parser_state; }

  std::vector<dynet::expr::Expression> loss;
  for (unsigned i = curr; i < next; ++i) {
    loss.push_back(scores_exprs[i]);
  }
  dynet::expr::Expression l =
    dynet::expr::pickneglogsoftmax(dynet::expr::concatenate(loss), corr - curr);
  float ret = dynet::as_scalar(cg.forward(l));
  cg.backward(l);
  trainer->update(1.f);
  return ret;
}

float SupervisedTrainer::train_partial_tree(const InputUnits& input_units,
                                            const ParseUnits& parse_units,
                                            dynet::Trainer* trainer,
                                            unsigned iter) {
  TransitionSystem & system = state_builder.parser_model->get_system();
  
  std::vector<unsigned> ref_heads, ref_deprels;
  parse_to_vector(parse_units, ref_heads, ref_deprels);

  ParserState * parser_state = state_builder.build();
  dynet::ComputationGraph cg;
  parser_state->new_graph(cg);
  parser_state->initialize(cg, input_units);

  unsigned len = input_units.size();
  TransitionState transition_state(len);
  transition_state.initialize(input_units);

  unsigned illegal_action = system.num_actions();
  unsigned n_actions = 0;

  std::vector<dynet::expr::Expression> loss;
  while (!transition_state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(transition_state, valid_actions);

    dynet::expr::Expression score_exprs = parser_state->get_scores();
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    unsigned action = 0;
    unsigned best_gold_action = illegal_action;
    unsigned worst_gold_action = illegal_action;
    unsigned best_non_gold_action = illegal_action;

    auto payload = ParserState::get_best_action(scores, valid_actions);
    action = payload.first;
    std::vector<float> costs; // the larger, the better
    system.get_transition_costs(transition_state, valid_actions, ref_heads, ref_deprels, costs);

    float gold_action_cost = (*std::max_element(costs.begin(), costs.end()));
    float action_cost = 0.f;
    float best_gold_action_score = -1e10, worst_gold_action_score = 1e10, best_non_gold_action_score = -1e10;
    for (unsigned i = 0; i < valid_actions.size(); ++i) {
      unsigned act = valid_actions[i];
      float s = scores[act];
      if (costs[i] == gold_action_cost) {
        if (best_gold_action_score < s) { best_gold_action_score = s; best_gold_action = act; }
        if (worst_gold_action_score > s) { worst_gold_action_score = s; worst_gold_action = act; }
      } else {
        if (best_non_gold_action_score < s) { best_non_gold_action_score = s; best_non_gold_action = act; }
      }
      if (act == action) { action_cost = costs[i]; }
    }
    if (gold_action_cost != action_cost) {
      action = best_gold_action;
    }

    add_loss_one_step(score_exprs, best_gold_action, worst_gold_action, best_non_gold_action, loss);

    system.perform_action(transition_state, action);
    parser_state->perform_action(action, cg, transition_state);
    n_actions++;
  }
  delete parser_state;

  float ret = 0.f;
  if (loss.size() > 0) {
    dynet::expr::Expression l = dynet::expr::sum(loss);
    ret = dynet::as_scalar(cg.forward(l));
    cg.backward(l);
    trainer->update(1.f);
  }
  return ret;
}
