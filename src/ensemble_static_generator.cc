#include "ensemble_static_generator.h"
#include "logging.h"
#include <fstream>
#include "tree.h"
#include "math_utils.h"

po::options_description EnsembleStaticDataGenerator::get_options() {
  po::options_description cmd("Supervised dynamic ensemble options");
  cmd.add_options()
    ("static_ensemble_method", po::value<std::string>()->default_value("prob"), "Ensemble methods [prob|logits_mean|logits_sum]")
    ("static_ensemble_n_sample", po::value<unsigned>()->default_value(1), "The number of sample.")
    ("static_ensemble_rollin", po::value<std::string>()->default_value("expert"), "The type of rollin policy [expert|egreedy|boltzmann].")
    ("static_ensemble_egreedy_epsilon", po::value<float>()->default_value(0.1f), "The epsilon for epsilon-greedy policy.")
    ("static_ensemble_boltzmann_temperature", po::value<float>()->default_value(1.f), "The epsilon for epsilon-greedy policy.")
    ;
  return cmd;
}

EnsembleStaticDataGenerator::EnsembleStaticDataGenerator(const po::variables_map & conf,
                                                         std::vector<ParserStateBuilder*>& pretrained_state_builders) :
  pretrained_state_builders(pretrained_state_builders) {
  n_pretrained = pretrained_state_builders.size();
  _INFO << "GEN:: number of parsers: " << n_pretrained;

  std::string ensemble_method_name = conf["static_ensemble_method"].as<std::string>();
  if (ensemble_method_name == "prob") {
    ensemble_method = kProbability;
  } else if (ensemble_method_name == "logits_mean") {
    ensemble_method = kLogitsMean;
  } else {
    ensemble_method = kLogitsSum;
  }
  _INFO << "GEN:: ensemble method: " << ensemble_method_name;

  n_sample = conf["static_ensemble_n_sample"].as<unsigned>();
  _INFO << "GEN:: generate # samples " << n_sample;

  std::string rollin_name = conf["static_ensemble_rollin"].as<std::string>();
  if (rollin_name == "egreedy") {
    rollin_type = kEpsilonGreedy;
  } else if (rollin_name == "boltzmann") {
    rollin_type = kBoltzmann;
  } else if (rollin_name == "expert") {
    rollin_type = kExpert;
  } else {
    _ERROR << "GEN:: Unknown oracle :" << rollin_name;
  }
  
  if (rollin_type == kEpsilonGreedy) {
    epsilon = conf["static_ensemble_egreedy_epsilon"].as<float>();
    _INFO << "GEN:: epsilon for egreedy policy: " << epsilon;
  } else if (rollin_type == kBoltzmann) {
    temperature = conf["static_ensemble_boltzmann_temperature"].as<float>();
    _INFO << "GEN:: temperature for boltzmann policy: " << temperature;
  }
}

void EnsembleStaticDataGenerator::generate(const po::variables_map & conf,
                                           Corpus & corpus,
                                           const std::string & output,
                                           bool allow_nonprojective) {
  _INFO << "GEN:: start lstm-parser supervised ensemble training data generation.";
  TransitionSystem & system = pretrained_state_builders[0]->system;

  unsigned logc = 0;
  unsigned report_stops = conf["report_stops"].as<unsigned>();
  unsigned n_devel = corpus.n_devel;

  std::ofstream ofs(output);
  ofs << "num_actions=" << system.num_actions() << std::endl;
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse_units = corpus.devel_parses[sid];
    if (!allow_nonprojective && DependencyUtils::is_non_projective(parse_units)) {
      _INFO << "ENS_STAT:: sentence " << sid << " is nonprojective.";
      continue;
    }

    for (unsigned n = 0; n < n_sample; ++n) {
      std::vector<unsigned> ref_heads, ref_deprels;
      parse_to_vector(parse_units, ref_heads, ref_deprels);
      std::vector<unsigned> gold_actions;
      if (rollin_type == kExpert) {
        system.get_oracle_actions(ref_heads, ref_deprels, gold_actions);
      }

      dynet::ComputationGraph cg;
      unsigned len = input_units.size();
      TransitionState transition_state(len);
      transition_state.initialize(input_units);

      std::vector<ParserState*> ensembled_parser_states(n_pretrained, nullptr);
      for (unsigned i = 0; i < n_pretrained; ++i) {
        ensembled_parser_states[i] = pretrained_state_builders[i]->build();
        ensembled_parser_states[i]->new_graph(cg);
        ensembled_parser_states[i]->initialize(cg, input_units);
      }

      ofs << sid << std::endl;
      unsigned n_actions = 0;
      while (!transition_state.terminated()) {
        std::vector<unsigned> valid_actions;
        system.get_valid_actions(transition_state, valid_actions);

        std::vector<float> ensembled_scores(system.num_actions(), 0.f);
        for (ParserState* ensembled_parser_state : ensembled_parser_states) {
          dynet::expr::Expression ensembled_score_exprs = ensembled_parser_state->get_scores();
          std::vector<float> ensembled_score = dynet::as_vector(cg.get_value(ensembled_score_exprs));
          if (ensemble_method == kProbability) { softmax_inplace(ensembled_score); }
          for (unsigned i = 0; i < ensembled_score.size(); ++i) {
            ensembled_scores[i] += ensembled_score[i];
          }
        }
        if (ensemble_method == kProbability || ensemble_method == kLogitsMean) {
          for (unsigned i = 0; i < ensembled_scores.size(); ++i) {
            ensembled_scores[i] /= n_pretrained;
          }
        }

        unsigned action = UINT_MAX;
        if (rollin_type == kExpert) {
          action = gold_actions[n_actions];
        } else if (rollin_type == kEpsilonGreedy) {
          float seed = dynet::rand01();
          if (seed < epsilon) {
            action = valid_actions[dynet::rand0n(valid_actions.size())];
          } else {
            auto payload = ParserState::get_best_action(ensembled_scores, valid_actions);
            action = payload.first;
          }
        } else {
          std::vector<float> valid_prob;
          for (unsigned act : valid_actions) {
            valid_prob.push_back(ensembled_scores[act] / temperature);
          }
          softmax_inplace(valid_prob);
          unsigned index = distribution_sample(valid_prob, (*dynet::rndeng));
          action = valid_actions[index];
        }

        if (ensemble_method == kLogitsMean || ensemble_method == kLogitsSum) {
          softmax_inplace(ensembled_scores);
        }
        ofs << action;
        for (float s : ensembled_scores) { ofs << " " << s; }
        ofs << std::endl;
        system.perform_action(transition_state, action);
        for (ParserState * ensembled_parser_state : ensembled_parser_states) {
          ensembled_parser_state->perform_action(action, cg, transition_state);
        }
        n_actions++;
      }
      ofs << std::endl;
      
      for (ParserState * parser_state : ensembled_parser_states) { delete parser_state; }
    }

    ++logc;
    if (logc % report_stops == 0) {
      float epoch = (float(logc) / n_devel);
      _INFO << "GEN:: finished " << epoch << " data.";
    }
  }
}
