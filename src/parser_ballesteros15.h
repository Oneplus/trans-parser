#ifndef PARSER_BALLESTEROS15_H
#define PARSER_BALLESTEROS15_H

#include "parser.h"
#include "corpus.h"
#include "system.h"
#include "dynet/lstm.h"
#include "dynet_layer/layer.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct Ballesteros15ParserModel : public ParserModel {
  dynet::CoupledLSTMBuilder fwd_ch_lstm;
  dynet::CoupledLSTMBuilder bwd_ch_lstm;
  dynet::CoupledLSTMBuilder s_lstm;
  dynet::CoupledLSTMBuilder q_lstm;
  dynet::CoupledLSTMBuilder a_lstm;

  SymbolEmbedding char_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;
  SymbolEmbedding act_emb;
  SymbolEmbedding rel_emb;

  Merge3Layer merge_input;  // merge (2 * word, pos, preword)
  Merge3Layer merge;        // merge (s_lstm, q_lstm, a_lstm)
  Merge3Layer composer;     // compose (head, modifier, relation)
  DenseLayer scorer;

  dynet::Parameter p_action_start;  // start of action
  dynet::Parameter p_buffer_guard;  // end of buffer
  dynet::Parameter p_stack_guard;   // end of stack
  dynet::Parameter p_word_start_guard;
  dynet::Parameter p_word_end_guard;
  dynet::Parameter p_root_word;
  dynet::Expression action_start;
  dynet::Expression buffer_guard;
  dynet::Expression stack_guard;
  dynet::Expression word_start_guard;
  dynet::Expression word_end_guard;
  dynet::Expression root_word;
  
  /// The reference
  const Embeddings & pretrained;

  /// The Configurations: useful for other models.
  unsigned size_c, dim_c, dim_w, size_p, dim_p, size_t, dim_t, size_l, dim_l, size_a, dim_a;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  Ballesteros15ParserModel(dynet::ParameterCollection & m,
                           unsigned size_c,
                           unsigned dim_c,
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
                           const Embeddings & pretrained);

  void new_graph(dynet::ComputationGraph & cg) override;

  std::vector<dynet::Expression> get_params() override;
};

struct Ballesteros15ParserState : public ParserState {
  struct ActionPerformer;

  Ballesteros15ParserModel & model;
  dynet::RNNPointer s_pointer;
  dynet::RNNPointer q_pointer;
  dynet::RNNPointer a_pointer;
  std::vector<dynet::Expression> stack;
  std::vector<dynet::Expression> buffer;
  ActionPerformer * performer;

  struct ActionPerformer {
    Ballesteros15ParserState * state;
    ActionPerformer(Ballesteros15ParserState * state) : state(state) {}
    virtual ~ActionPerformer() {}
    virtual void perform_action(const unsigned& action,
                                dynet::ComputationGraph& cg) = 0;
  };

  struct ArcEagerPerformer : public ActionPerformer {
    ArcEagerPerformer(Ballesteros15ParserState * state) : ActionPerformer(state) {}
    ~ArcEagerPerformer() {}
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg) override;
  };

  struct ArcStandardPerformer : public ActionPerformer {
    ArcStandardPerformer(Ballesteros15ParserState * state) : ActionPerformer(state) {}
    ~ArcStandardPerformer() {}
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg) override;
  };

  struct ArcHybridPerformer : public ActionPerformer {
    ArcHybridPerformer(Ballesteros15ParserState * state) : ActionPerformer(state) {}
    ~ArcHybridPerformer() {}
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg) override;
  };

  struct SwapPerformer : public ActionPerformer {
    SwapPerformer(Ballesteros15ParserState * state) : ActionPerformer(state) {}
    ~SwapPerformer() {}
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg) override;
  };

  Ballesteros15ParserState(Ballesteros15ParserModel & model);
  ~Ballesteros15ParserState() { if (performer != nullptr) { delete performer; } }

  void new_graph(dynet::ComputationGraph& cg) override;

  void initialize(dynet::ComputationGraph& cg,
                  const InputUnits& input) override;

  void perform_action(const unsigned & action,
                      dynet::ComputationGraph & cg,
                      const TransitionState & state) override;

  ParserState * copy() override;

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::Expression get_scores() override;

  std::vector<dynet::Expression> get_params() override;
};

struct Ballesteros15ParserStateBuilder : public ParserStateBuilder {
  Ballesteros15ParserModel * parser_model;

  Ballesteros15ParserStateBuilder(const po::variables_map & conf,
                                  dynet::ParameterCollection & model,
                                  TransitionSystem & system,
                                  const Corpus & corpus,
                                  const Embeddings & pretrained);

  ParserState * build() override;
};

#endif  //  end for PARSER_H
