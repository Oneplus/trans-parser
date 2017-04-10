#ifndef PARSER_KIPERWASSER_H
#define PARSER_KIPERWASSER_H

#include "parser.h"
#include "layer.h"
#include "corpus.h"
#include "state.h"
#include "system.h"
#include <vector>
#include <unordered_map>

struct Kiperwasser16ParserModel : public ParserModel {
  dynet::LSTMBuilder fwd_lstm;
  dynet::LSTMBuilder bwd_lstm;
  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;

  Merge3Layer merge_input;
  Merge4Layer merge;        // merge (s2, s1, s0, n0)
  DenseLayer scorer;

  dynet::Parameter p_empty;
  dynet::Parameter p_fwd_guard;   // start of fwd
  dynet::Parameter p_bwd_guard;   // end of bwd
  dynet::expr::Expression empty;
  dynet::expr::Expression fwd_guard;
  dynet::expr::Expression bwd_guard;

  const Embeddings & pretrained;

  unsigned size_w, dim_w, size_p, dim_p, size_t, dim_t, size_a;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  Kiperwasser16ParserModel(dynet::Model& m,
                           unsigned size_w,  //
                           unsigned dim_w,   // word size, word dim
                           unsigned size_p,  //
                           unsigned dim_p,   // pos size, pos dim
                           unsigned size_t,  //
                           unsigned dim_t,   // pword size, pword dim
                           unsigned size_a,  //
                           unsigned n_layers,
                           unsigned dim_lstm_in,
                           unsigned dim_hidden,
                           TransitionSystem & system,
                           const Embeddings & pretrained);

  void new_graph(dynet::ComputationGraph & cg) override;

  std::vector<dynet::expr::Expression> get_params() override;
};

struct Kiperwasser16ParserState : public ParserState {
  struct FeatureExtractor;

  Kiperwasser16ParserModel & model;
  std::vector<dynet::expr::Expression> encoded;
  dynet::expr::Expression f0;
  dynet::expr::Expression f1;
  dynet::expr::Expression f2;
  dynet::expr::Expression f3;
  FeatureExtractor * extractor;

  struct FeatureExtractor {
    Kiperwasser16ParserState * hook;
    FeatureExtractor(Kiperwasser16ParserState * hook) : hook(hook) {}
    virtual ~FeatureExtractor() {}
    // should do after sys.perform_action
    virtual void extract(const TransitionState & state) = 0;
  };

  struct ArcEagerExtractor : public FeatureExtractor {
    ArcEagerExtractor(Kiperwasser16ParserState * hook) : FeatureExtractor(hook) {}
    ~ArcEagerExtractor() {}
    void extract(const TransitionState & state) override;
  };

  struct ArcStandardExtractor : public FeatureExtractor {
    ArcStandardExtractor(Kiperwasser16ParserState * hook) : FeatureExtractor(hook) {}
    ~ArcStandardExtractor() {}
    void extract(const TransitionState & state) override;
  };

  struct ArcHybridExtractor : public FeatureExtractor {
    ArcHybridExtractor(Kiperwasser16ParserState * hook) : FeatureExtractor(hook) {}
    ~ArcHybridExtractor() {}
    void extract(const TransitionState & state) override;
  };

  struct SwapExtractor : public FeatureExtractor {
    SwapExtractor(Kiperwasser16ParserState * hook) : FeatureExtractor(hook) {}
    ~SwapExtractor() {}
    void extract(const TransitionState & state) override;
  };

  Kiperwasser16ParserState(Kiperwasser16ParserModel & model);

  ~Kiperwasser16ParserState() { if (extractor != nullptr) { delete extractor; } }

  void new_graph(dynet::ComputationGraph & cg) override;

  void initialize(dynet::ComputationGraph& cg,
                  const InputUnits & input) override;

  void perform_action(const unsigned & action,
                      dynet::ComputationGraph& cg,
                      const TransitionState & state) override;

  ParserState * copy() override;

  dynet::expr::Expression get_scores() override;

  std::vector<dynet::expr::Expression> get_params() override;
};

struct Kiperwasser16ParserStateBuilder : public ParserStateBuilder {
  Kiperwasser16ParserModel * parser_model;

  Kiperwasser16ParserStateBuilder(const po::variables_map & conf,
                                  dynet::Model & model,
                                  TransitionSystem & system,
                                  const Corpus & corpus,
                                  const Embeddings & pretrained);

  ParserState * build() override;
};


#endif  //  end for PARSER_BILSTM_H