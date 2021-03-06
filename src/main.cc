#include <iostream>
#include <fstream>
#include <set>
#include "dynet/init.h"
#include "corpus.h"
#include "logging.h"
#include "parser_builder.h"
#include "noisify.h"
#include "system_builder.h"
#include "evaluate.h"
#include "train_supervised.h"
#include "sys_utils.h"
#include "trainer_utils.h"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description general("Transition-based dependency parser.");
  general.add_options()
    ("train,t", "Use to specify to perform training.")
    ("word_list", po::value<std::string>(), "(Optional) The path to the word list.")
    ("pos_list", po::value<std::string>(), "(Optional) The path to the pos list.")
    ("deprel_list", po::value<std::string>(), "(Optional) The path to the deprel list.")
    ("architecture", po::value<std::string>()->default_value("d15"), "The architecture [dyer15, ballesteros15, kiperwasser16].")
    ("training_data,T", po::value<std::string>()->required(), "The path to the training data.")
    ("devel_data,d", po::value<std::string>()->required(), "The path to the development data.")
    ("pretrained,w", po::value<std::string>(), "The path to the word embedding.")
    ("model,m", po::value<std::string>(), "The path to the model.")
    ("layers", po::value<unsigned>()->default_value(2), "The number of layers in LSTM.")
    ("char_dim", po::value<unsigned>()->default_value(50), "The dimension of char.")
    ("word_dim", po::value<unsigned>()->default_value(32), "number of LSTM layers.")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dim, set it as 0 to disable POS.")
    ("pretrained_dim", po::value<unsigned>()->default_value(100), "Pretrained input dimension.")
    ("action_dim", po::value<unsigned>()->default_value(20), "The dimension for action.")
    ("label_dim", po::value<unsigned>()->default_value(20), "The dimension for label.")
    ("lstm_input_dim", po::value<unsigned>()->default_value(100), "The dimension for lstm input.")
    ("hidden_dim", po::value<unsigned>()->default_value(100), "The dimension for hidden unit.")
    ("dropout", po::value<float>()->default_value(0.f), "The dropout rate.")
    ("max_iter", po::value<unsigned>()->default_value(10), "The maximum number of iteration.")
    ("report_stops", po::value<unsigned>()->default_value(100), "The reporting stops")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "The evaluation stops")
    ("evaluate_skips", po::value<unsigned>()->default_value(0), "skip evaluation on the first n round.")
    ("external_eval", po::value<std::string>()->default_value("python ./script/eval.py"), "config the path for evaluation script")
    ("lambda", po::value<float>()->default_value(0.), "The L2 regularizer, should not set in --dynet-l2.")
    ("output", po::value<std::string>(), "The path to the output file.")
    ("beam_size", po::value<unsigned>(), "The beam size.")
    ("partial", po::value<bool>()->default_value(false), "The input data contains partial annotation.")
    ("verbose,v", "Details logging.")
    ("help,h", "show help information")
    ;

  po::options_description system_opt = TransitionSystemBuilder::get_options();
  po::options_description noisify_opt = Noisifier::get_options();
  po::options_description optimizer_opt = get_optimizer_options();
  po::options_description supervise_opt = SupervisedTrainer::get_options();

  po::options_description cmd("Allowed options");
  cmd.add(general)
    .add(system_opt)
    .add(noisify_opt)
    .add(optimizer_opt)
    .add(supervise_opt)
    ;

  po::store(po::parse_command_line(argc, argv, cmd), conf);
  if (conf.count("help")) {
    std::cerr << cmd << std::endl;
    exit(1);
  }
  init_boost_log(conf.count("verbose") > 0);
  if (!conf.count("training_data")) {
    std::cerr << "Please specify --training_data (-T), even in test" << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, false);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, conf);

  std::string model_name;
  if (conf.count("train")) {
    if (conf.count("model")) {
      model_name = conf["model"].as<std::string>();
    } else {
      std::string prefix("parser_l2r.model");
      model_name = get_model_name(conf, prefix);
    }
    _INFO << "Main:: write parameters to: " << model_name;
  } else {
    model_name = conf["model"].as<std::string>();
    _INFO << "Main:: evaluating model from: " << model_name;
  }

  bool allow_partial_tree = conf["partial"].as<bool>();
  Corpus corpus;
  if (conf.count("word_list")) {
    corpus.load_word_list(conf["word_list"].as<std::string>());
  }
  if (conf.count("pos_list")) {
    corpus.load_pos_list(conf["pos_list"].as<std::string>());
  }
  if (conf.count("deprel_list")) {
    corpus.load_deprel_list(conf["deprel_list"].as<std::string>());
  }
  std::unordered_map<unsigned, std::vector<float>> pretrained;
  if (conf.count("pretrained")) {
    corpus.load_word_embeddings(conf["pretrained"].as<std::string>(),
                                conf["pretrained_dim"].as<unsigned>(),
                                pretrained);
  } else {
    corpus.load_empty_embeddings(conf["pretrained_dim"].as<unsigned>(),
                                 pretrained);
  }

  corpus.load_training_data(conf["training_data"].as<std::string>(), allow_partial_tree);
  corpus.stat();
  corpus.get_vocabulary_and_word_count();

  _INFO << "Main:: after loading pretrained embedding, size(vocabulary)=" << corpus.word_map.size();

  dynet::ParameterCollection model;
  TransitionSystem* sys = TransitionSystemBuilder(corpus).build(conf);
  bool allow_non_projective = TransitionSystemBuilder::allow_nonprojective(conf);

  Noisifier noisifier(conf, corpus);
  ParserStateBuilder * state_builder = get_state_builder(conf, model, (*sys), corpus, pretrained);

  corpus.load_devel_data(conf["devel_data"].as<std::string>(), allow_partial_tree);
  _INFO << "Main:: after loading development data, size(vocabulary)=" << corpus.word_map.size();

  std::string output;
  if (conf.count("output")) {
    output = conf["output"].as<std::string>();
  } else {
    int pid = portable_getpid();
#ifdef _MSC_VER
    output = "parser_l2r.evaluator." + boost::lexical_cast<std::string>(pid);
#else
    output = "/tmp/parser_l2r.evaluator." + boost::lexical_cast<std::string>(pid);
#endif
  }
  _INFO << "Main:: write tmp file to: " << output;

  if (conf.count("train")) {
    SupervisedTrainer trainer(conf, noisifier, *state_builder);
    trainer.train(conf, corpus, model_name, output, allow_non_projective, allow_partial_tree);
  }

  dynet::load_dynet_model(model_name, (&model));
  if (conf.count("beam_size") && conf["beam_size"].as<unsigned>() > 1) {
    bool structure_test = (conf["supervised_objective"].as<std::string>() == "structure");
    beam_search(conf, corpus, *state_builder, output, structure_test);
  } else {
    evaluate(conf, corpus, *state_builder, output);
  }
  return 0;
}

