#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"


struct LayerI {
  bool trainable;
  LayerI(bool trainable) : trainable(trainable) {}
  void active_training()   { trainable = true;  }
  void inactive_training() { trainable = false; }
  // Initialize parameter
  virtual void new_graph(dynet::ComputationGraph& cg) = 0;
  virtual std::vector<dynet::expr::Expression> get_params() = 0;
};

struct SymbolEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_labels;

  SymbolEmbedding(dynet::Model& m, unsigned n, unsigned dim, bool trainable=true);
  SymbolEmbedding(dynet::Model& m, unsigned n, unsigned dim, float scale, bool trainable=true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression embed(unsigned label_id);
  dynet::expr::Expression embed(const std::vector<unsigned> & label_ids);
};

struct BinnedDistanceEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDistanceEmbedding(dynet::Model& m,
                          unsigned hidden,
                          unsigned n_bin=8,
                          bool trainable=true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression embed(int distance);
};

struct BinnedDurationEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDurationEmbedding(dynet::Model& m,
                          unsigned hidden,
                          unsigned n_bin=8,
                          bool trainable=true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression embed(unsigned dur);
};

typedef std::pair<dynet::expr::Expression, dynet::expr::Expression> BiRNNOutput;

template<typename Builder>
struct RNNLayer : public LayerI {
  unsigned n_items;
  Builder rnn;
  dynet::Parameter p_guard;
  dynet::expr::Expression guard;
  bool reversed;

  RNNLayer(dynet::Model& model,
           unsigned n_layers,
           unsigned dim_input,
           unsigned dim_hidden,
           bool rev=false,
           bool trainable=true) :
    LayerI(trainable),
    n_items(0),
    rnn(n_layers, dim_input, dim_hidden, &model),
    p_guard(model.add_parameters({ dim_input, 1 })),
    reversed(rev) {
  }

  void add_inputs(const std::vector<dynet::expr::Expression>& inputs) {
    n_items = inputs.size();
    rnn.start_new_sequence();
    rnn.add_input(guard);
    if (reversed) { 
      for (int i = n_items - 1; i >= 0; --i) { rnn.add_input(inputs[i]); } 
    } else {
      for (unsigned i = 0; i < n_items; ++i) { rnn.add_input(inputs[i]); }
    }
  }

  dynet::expr::Expression get_output(dynet::ComputationGraph* hg,
                                   int index) {
    if (reversed) { return rnn.get_h(dynet::RNNPointer(n_items - index)).back(); }
    return rnn.get_h(dynet::RNNPointer(index + 1)).back();
  }

  void get_outputs(dynet::ComputationGraph* hg, 
                   std::vector<dynet::expr::Expression>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) { outputs[i] = get_output(hg, i); }
  }

  void new_graph(dynet::ComputationGraph& hg) {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    rnn.new_graph(hg);
    guard = dynet::expr::parameter(hg, p_guard);
  }
  void set_dropout(float& rate) { rnn.set_dropout(rate); }
  void disable_dropout() { rnn.disable_dropout(); }
};

template<typename Builder>
struct BidirectionalRNNLayer : public LayerI {
  unsigned n_items;
  Builder fw_rnn;
  Builder bw_rnn;
  dynet::Parameter p_fw_guard;
  dynet::Parameter p_bw_guard;
  dynet::expr::Expression fw_guard;
  dynet::expr::Expression bw_guard;

  BidirectionalRNNLayer(dynet::Model& model,
                        unsigned n_layers,
                        unsigned dim_input,
                        unsigned dim_hidden) :
    n_items(0),
    fw_rnn(n_layers, dim_input, dim_hidden, &model),
    bw_rnn(n_layers, dim_input, dim_hidden, &model),
    p_fw_guard(model.add_parameters({ dim_input, 1 })),
    p_bw_guard(model.add_parameters({ dim_input, 1 })) {
  }

  void add_inputs(const std::vector<dynet::expr::Expression>& inputs) {
    n_items = inputs.size();
    fw_rnn.start_new_sequence();
    bw_rnn.start_new_sequence();

    fw_rnn.add_input(fw_guard);
    for (unsigned i = 0; i < n_items; ++i) {
      fw_rnn.add_input(inputs[i]);
      bw_rnn.add_input(inputs[n_items - i - 1]);
    }
    bw_rnn.add_input(bw_guard);
  }

  BiRNNOutput get_output(int index) {
    return std::make_pair(
      fw_rnn.get_h(dynet::RNNPointer(index + 1)).back(),
      bw_rnn.get_h(dynet::RNNPointer(n_items - index - 1)).back());
  }

  void get_outputs(std::vector<BiRNNOutput>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) {
      outputs[i] = get_output(i);
    }
  }

  void new_graph(dynet::ComputationGraph& hg) {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    fw_rnn.new_graph(hg);
    bw_rnn.new_graph(hg);
    fw_guard = dynet::expr::parameter(hg, p_fw_guard);
    bw_guard = dynet::expr::parameter(hg, p_bw_guard);
  }

  std::vector<dynet::expr::Expression> get_params() {
    std::vector<dynet::expr::Expression> ret = { fw_guard, bw_guard };
    return ret;
  }

  void set_dropout(float& rate) {
    fw_rnn.set_dropout(rate);
    bw_rnn.set_dropout(rate);
  }

  void disable_dropout() {
    fw_rnn.disable_dropout();
    bw_rnn.disable_dropout();
  }
};


struct CNNLayer : public LayerI {
  std::vector<std::vector<dynet::Parameter>> p_filters;
  std::vector<std::vector<dynet::Parameter>> p_biases;
  std::vector<std::vector<dynet::expr::Expression>> filters;
  std::vector<std::vector<dynet::expr::Expression>> biases;
  std::vector<std::vector<dynet::expr::Expression>> h;
  std::vector<std::pair<unsigned, unsigned>> filters_info;
  dynet::Parameter p_W;
  dynet::expr::Expression W;
  dynet::expr::Expression padding;
  unsigned input_dim;

  CNNLayer(dynet::Model& m,
           unsigned input_dim,
           unsigned output_dim,
           const std::vector<std::pair<unsigned, unsigned>>& info,
           bool trainable=true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const std::vector<dynet::expr::Expression>& exprs);
};

struct SoftmaxLayer : public LayerI {
  dynet::Parameter p_B, p_W;
  dynet::expr::Expression B, W;

  SoftmaxLayer(dynet::Model& model,
               unsigned dim_input,
               unsigned dim_output,
               bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr);
};

struct DenseLayer : public LayerI {
  dynet::Parameter p_W, p_B;
  dynet::expr::Expression W, B;

  DenseLayer(dynet::Model& model,
             unsigned dim_input,
             unsigned dim_output,
             bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr);
};

struct Merge2Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2;
  dynet::expr::Expression B, W1, W2;

  Merge2Layer(dynet::Model& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_output,
              bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr1,
                                     const dynet::expr::Expression& expr2);
};


struct Merge3Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3;
  dynet::expr::Expression B, W1, W2, W3;

  Merge3Layer(dynet::Model& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_output,
              bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr1,
                                     const dynet::expr::Expression& expr2,
                                     const dynet::expr::Expression& expr3);
};

struct Merge4Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4;
  dynet::expr::Expression B, W1, W2, W3, W4;

  Merge4Layer(dynet::Model& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_output,
              bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr1,
                                     const dynet::expr::Expression& expr2,
                                     const dynet::expr::Expression& expr3,
                                     const dynet::expr::Expression& expr4);
};

struct Merge5Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5;
  dynet::expr::Expression B, W1, W2, W3, W4, W5;

  Merge5Layer(dynet::Model& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_output,
              bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr1,
                                     const dynet::expr::Expression& expr2,
                                     const dynet::expr::Expression& expr3,
                                     const dynet::expr::Expression& expr4,
                                     const dynet::expr::Expression& expr5);
};

struct Merge6Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5, p_W6;
  dynet::expr::Expression B, W1, W2, W3, W4, W5, W6;

  Merge6Layer(dynet::Model& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_input6,
              unsigned dim_output,
              bool trainable=true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::expr::Expression> get_params() override;
  dynet::expr::Expression get_output(const dynet::expr::Expression& expr1,
                                     const dynet::expr::Expression& expr2,
                                     const dynet::expr::Expression& expr3,
                                     const dynet::expr::Expression& expr4,
                                     const dynet::expr::Expression& expr5,
                                     const dynet::expr::Expression& expr6);
};

struct SegUniEmbedding {
  // uni-directional segment embedding.
  dynet::Parameter p_h0;
  dynet::LSTMBuilder builder;
  std::vector<std::vector<dynet::expr::Expression>> h;
  unsigned len;

  explicit SegUniEmbedding(dynet::Model& m,
                           unsigned n_layers,
                           unsigned lstm_input_dim,
                           unsigned seg_dim);
  void construct_chart(dynet::ComputationGraph& cg,
                       const std::vector<dynet::expr::Expression>& c,
                       int max_seg_len=0);
  const dynet::expr::Expression& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};

struct SegBiEmbedding {
  typedef std::pair<dynet::expr::Expression, dynet::expr::Expression> ExpressionPair;
  SegUniEmbedding fwd, bwd;
  std::vector<std::vector<ExpressionPair>> h;
  unsigned len;

  explicit SegBiEmbedding(dynet::Model& m,
                          unsigned n_layers,
                          unsigned lstm_input_dim,
                          unsigned seg_dim);
  void construct_chart(dynet::ComputationGraph& cg,
                       const std::vector<dynet::expr::Expression>& c,
                       int max_seg_len = 0);
  const ExpressionPair& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};

struct SegConcateEmbedding {
  dynet::Parameter p_W;
  dynet::Parameter p_b;
  std::vector<float> paddings;
  std::vector<std::vector<dynet::expr::Expression>> h;
  unsigned len;
  unsigned input_dim;
  unsigned max_seg_len;

  explicit SegConcateEmbedding(dynet::Model& m,
                               unsigned input_dim,
                               unsigned output_dim,
                               unsigned max_seg_len);
  void construct_chart(dynet::ComputationGraph& cg,
                       const std::vector<dynet::expr::Expression>& c,
                       int max_seg_len = 0);
  const dynet::expr::Expression& operator()(unsigned i, unsigned j) const;
};

#endif  //  end for LAYER_H
