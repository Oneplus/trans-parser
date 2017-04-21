#include "trainer_utils.h"
#include "train_supervised_ensemble.h"
#include "tree.h"
#include "logging.h"
#include "evaluate.h"
#include "math_utils.h"

po::options_description SupervisedEnsembleTrainer::get_options() {
  po::options_description cmd("Supervised ensemble options");
  cmd.add_options()
    ("ensemble_rollin", po::value<std::string>()->default_value("egreedy"), "The type of rollin policy [expert|egreedy].")
    ("ensemble_objective", po::value<std::string>()->default_value("crossentropy"), "The learning objective [crossentropy|sparse_crossentropy]")
    ("ensemble_egreedy_epsilon", po::value<float>()->default_value(0.1f), "The epsilon for epsilon-greedy policy.")
    ;
  return cmd;
}

SupervisedEnsembleTrainer::SupervisedEnsembleTrainer(const po::variables_map& conf,
                                                     const Noisifier& noisifier,
                                                     ParserStateBuilder & state_builder,
                                                     std::vector<ParserStateBuilder *>& pretrained_state_builders) :
  state_builder(state_builder),
  noisifier(noisifier),
  pretrained_state_builders(pretrained_state_builders),
  n_pretrained(pretrained_state_builders.size()) {
  lambda_ = conf["lambda"].as<float>();
  _INFO << "ENS:: lambda = " << lambda_;

  if (conf["ensemble_rollin"].as<std::string>() == "egreedy") {
    rollin_type = kEpsilonGreedy;
  } else if (conf["ensemble_rollin"].as<std::string>() == "expert") {
    rollin_type = kExpert;
  } else {
    _ERROR << "Unknown oracle :" << conf["ensemble_rollin"].as<std::string>();
  }

  std::string objective_name = conf["ensemble_objective"].as<std::string>();
  if (objective_name == "crossentropy") {
    objective_type = kCrossEntropy;
  } else if (objective_name == "sparse_crossentropy") {
    objective_type = kSparseCrossEntropy;
  } else {
    _ERROR << "ENS:: unknown objective" << objective_name;
    exit(1);
  }                                                                       
  _INFO << "ENS:: learning objective " << objective_name;
  
  if (rollin_type == kEpsilonGreedy) {
    epsilon = conf["ensemble_egreedy_epsilon"].as<float>();
    _INFO << "ENS:: epsilon for egreedy policy: " << epsilon;
  }
}

void SupervisedEnsembleTrainer::train(const po::variables_map& conf,
                                      Corpus& corpus,
                                      const std::string& name,
                                      const std::string& output,
                                      bool allow_nonprojective,
                                      bool allow_partial_tree) {
  dynet::Model& model = state_builder.model;
  _INFO << "ENS:: start lstm-parser supervised ensemble training.";

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

  _INFO << "ENS:: will stop after " << max_iter << " iterations.";
  for (unsigned iter = 0; iter < max_iter; ++iter) {
    llh = 0;
    _INFO << "ENS:: start training iteration #" << iter << ", shuffled.";
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits& input_units = corpus.training_inputs[sid];
      const ParseUnits& parse_units = corpus.training_parses[sid];
      
      noisifier.noisify(input_units);
      float lp = train_full_tree(input_units, parse_units, trainer, iter);

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
        float f = evaluate(conf, corpus, state_builder, output);
        if (f > best_f) {
          best_f = f;
          _INFO << "SUP:: new best record achieved: " << best_f << ", saved.";
          dynet::save_dynet_model(name, (&model));
        }
      }
    }

    _INFO << "SUP:: end of iter #" << iter << " loss " << llh;
    float f = evaluate(conf, corpus, state_builder, output);
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

void SupervisedEnsembleTrainer::add_loss_one_step(dynet::expr::Expression & score_expr,
                                                  const std::vector<unsigned> & valid_actions,
                                                  const std::vector<float> & probs,
                                                  std::vector<dynet::expr::Expression> & loss) {
  TransitionSystem & system = state_builder.system;
  unsigned illegal_action = system.num_actions();

  if (objective_type == kSparseCrossEntropy) {
    auto best = ParserState::get_best_action(probs, valid_actions);
    loss.push_back(dynet::expr::pickneglogsoftmax(score_expr, best.first));
  } else {
    unsigned n_probs = probs.size();
    loss.push_back(-dynet::dot_product(
      dynet::expr::input(*score_expr.pg, { n_probs }, probs),
      dynet::expr::log_softmax(score_expr)
    ));
  }
}

float SupervisedEnsembleTrainer::train_full_tree(const InputUnits& input_units,
                                                 const ParseUnits& parse_units,
                                                 dynet::Trainer* trainer,
                                                 unsigned iter) {
  TransitionSystem & system = state_builder.system;

  std::vector<unsigned> ref_heads, ref_deprels;
  parse_to_vector(parse_units, ref_heads, ref_deprels);
  std::vector<unsigned> gold_actions;
  if (rollin_type == kExpert) {
    system.get_oracle_actions(ref_heads, ref_deprels, gold_actions);
  }

  ParserState * parser_state = state_builder.build();
  dynet::ComputationGraph cg;
  parser_state->new_graph(cg);
  parser_state->initialize(cg, input_units);

  std::vector<ParserState*> ensembled_parser_states(n_pretrained);
  for (unsigned i = 0; i < n_pretrained; ++i) {
    ensembled_parser_states[i] = pretrained_state_builders[i]->build();
    ensembled_parser_states[i]->new_graph(cg);
    ensembled_parser_states[i]->initialize(cg, input_units);
  }

  unsigned len = input_units.size();
  TransitionState transition_state(len);
  transition_state.initialize(input_units);

  unsigned n_actions = 0;
  std::vector<dynet::expr::Expression> loss;
  while (!transition_state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(transition_state, valid_actions);

    dynet::expr::Expression score_exprs = parser_state->get_scores();
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    std::vector<float> ensembled_scores(scores.size(), 0.f);
    for (ParserState* ensembled_parser_state : ensembled_parser_states) {
      dynet::expr::Expression ensembled_score_exprs = ensembled_parser_state->get_scores();
      std::vector<float> ensembled_score = dynet::as_vector(cg.get_value(ensembled_score_exprs));
      for (unsigned i = 0; i < ensembled_score.size(); ++i) {
        ensembled_scores[i] += ensembled_score[i];
      }
    }
    for (unsigned i = 0; i < ensembled_scores.size(); ++i) {
      ensembled_scores[i] /= n_pretrained;
    }
    softmax_inplace(ensembled_scores);

    add_loss_one_step(score_exprs, valid_actions, ensembled_scores, loss);

    unsigned action = UINT_MAX;
    if (rollin_type == kExpert) {
      action = gold_actions[n_actions];
    } else if (rollin_type == kEpsilonGreedy) {
      float seed = dynet::rand01();
      if (seed < epsilon) {
        action = valid_actions[dynet::rand0n(valid_actions.size())];
      } else {
        auto payload = ParserState::get_best_action(scores, valid_actions);
        action = payload.first;
      }
    }
    system.perform_action(transition_state, action);
    parser_state->perform_action(action, cg, transition_state);
    for (ParserState * ensembled_parser_state : ensembled_parser_states) {
      ensembled_parser_state->perform_action(action, cg, transition_state);
    }
    n_actions++;
  }
  float ret = 0.;
  if (loss.size() > 0) {
    std::vector<dynet::expr::Expression> all_params = parser_state->get_params();
    std::vector<dynet::expr::Expression> reg;
    for (auto e : all_params) { reg.push_back(dynet::expr::squared_norm(e)); }
    dynet::expr::Expression l = dynet::expr::sum(loss) + 0.5 * loss.size() * lambda_ * dynet::expr::sum(reg);
    ret = dynet::as_scalar(cg.incremental_forward(l));
    cg.backward(l);
    trainer->update(1.f);
  }
  delete parser_state;
  return ret;
}
