#include "train_supervised_ensemble_static.h"
#include "trainer_utils.h"
#include "train_supervised_ensemble_static.h"
#include "logging.h"
#include "tree.h"
#include "evaluate.h"
#include <random>

SupervisedEnsembleStaticTrainer::SupervisedEnsembleStaticTrainer(const po::variables_map & conf,
                                                                 const Noisifier & noisifier, 
                                                                 ParserStateBuilder & state_builder) : 
  state_builder(state_builder),
  noisifier(noisifier) {
  lambda_ = conf["lambda"].as<float>();
}

void SupervisedEnsembleStaticTrainer::train(const po::variables_map & conf, 
                                            CorpusWithActions & corpus,
                                            const std::string & name,
                                            const std::string & output,
                                            bool allow_nonprojective) {
  dynet::Model & model = state_builder.model;
  _INFO << "ENS_STAT:: start training static ensemble LSTM-parser.";

  dynet::Trainer* trainer = get_trainer(conf, model);
  unsigned max_iter = conf["max_iter"].as<unsigned>();

  float llh = 0.f, llh_in_batch = 0.f, best_f = 0.f;
  std::vector<unsigned> order;
  for (unsigned i = 0; i < corpus.training_actions.size(); ++i) {
    unsigned train_id = corpus.training_actions[i].train_id;
    const ParseUnits & parse_units = corpus.training_parses[train_id];

    if (!DependencyUtils::is_tree(parse_units)) { continue; }
    if (!allow_nonprojective && DependencyUtils::is_non_projective(parse_units)) { continue; }
    order.push_back(i);
  }

  float n_train = order.size();
  unsigned logc = 0;
  unsigned report_stops = conf["report_stops"].as<unsigned>();
  unsigned evaluate_stops = conf["evaluate_stops"].as<unsigned>();
  unsigned evaluate_skips = conf["evaluate_skips"].as<unsigned>();

  _INFO << "ENS_STAT:: will stop after " << max_iter << " iterations.";
  for (unsigned iter = 0; iter < max_iter; ++iter) {
    llh = 0;
    _INFO << "ENS_STAT:: start training iteration #" << iter << ", shuffled.";
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned aid : order) {
      const ActionUnits & action_units = corpus.training_actions[aid];
      unsigned sid = action_units.train_id;
      InputUnits & input_units = corpus.training_inputs[sid];
      const ParseUnits & parse_units = corpus.training_parses[sid];

      noisifier.noisify(input_units);
      float lp = train_full_tree(input_units, parse_units, action_units, trainer);

      llh += lp;
      llh_in_batch += lp;
      noisifier.denoisify(input_units);
    
      ++logc;
      if (logc % report_stops == 0) {
        float epoch = (float(logc) / n_train);
        _INFO << "ENS_STAT:: iter #" << iter << " (epoch " << epoch << ") loss " << llh_in_batch;
        llh_in_batch = 0.f;
      }
      if (iter >= evaluate_skips && logc % evaluate_stops == 0) {
        float f = evaluate(conf, corpus, state_builder, output);
        if (f > best_f) {
          best_f = f;
          _INFO << "ENS_STAT:: new best record achieved: " << best_f << ", saved.";
          dynet::save_dynet_model(name, (&model));
        }
      }
    }

    _INFO << "ENS_STAT:: end of iter #" << iter << " loss " << llh;
    float f = evaluate(conf, corpus, state_builder, output);
    if (f > best_f) {
      best_f = f;
      _INFO << "ENS_STAT:: new best record achieved: " << best_f << ", saved.";
      dynet::save_dynet_model(name, (&model));
    }
    trainer->update_epoch();
    trainer->status();
  }
  delete trainer;
}

float SupervisedEnsembleStaticTrainer::train_full_tree(const InputUnits & input_units,
                                                       const ParseUnits & parse_units,
                                                       const ActionUnits & action_units,
                                                       dynet::Trainer * trainer) {
  TransitionSystem & system = state_builder.system;
  std::vector<unsigned> ref_actions;
  for (unsigned i = 0; i < action_units.actions.size(); ++i) {
    ref_actions.push_back(action_units.actions[i].action);
  }
  ParserState * parser_state = state_builder.build();
  dynet::ComputationGraph cg;
  parser_state->new_graph(cg);
  parser_state->initialize(cg, input_units);

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

    add_loss_one_step(score_exprs, valid_actions, action_units.actions[n_actions].prob,
                      loss);

    unsigned action = action_units.actions[n_actions].action;
    system.perform_action(transition_state, action);
    parser_state->perform_action(action, cg, transition_state);
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

void SupervisedEnsembleStaticTrainer::add_loss_one_step(dynet::expr::Expression & score_expr,
                                                        const std::vector<unsigned>& valid_actions,
                                                        const std::vector<float>& probs,
                                                        std::vector<dynet::expr::Expression>& loss) {
  TransitionSystem & system = state_builder.system;
  unsigned illegal_action = system.num_actions();

  unsigned n_probs = probs.size();
  loss.push_back(-dynet::dot_product(
    dynet::expr::input(*score_expr.pg, { n_probs }, probs),
    dynet::expr::log_softmax(score_expr)
  ));
}
