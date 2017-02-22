#include "layer.h"

SymbolEmbedding::SymbolEmbedding(dynet::Model& m,
                                 unsigned n,
                                 unsigned dim,
                                 bool trainable) :
  LayerI(trainable),
  p_labels(m.add_lookup_parameters(n, { dim, 1 })) {
}

void SymbolEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::expr::Expression SymbolEmbedding::embed(unsigned label_id) {
  return (trainable ?
          dynet::expr::lookup((*cg), p_labels, label_id) :
          dynet::expr::const_lookup((*cg), p_labels, label_id));
}

BinnedDistanceEmbedding::BinnedDistanceEmbedding(dynet::Model& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins * 2, { dim, 1 })),
  max_bin(n_bins - 1) {
  BOOST_ASSERT(n_bins > 0);
}

void BinnedDistanceEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::expr::Expression BinnedDistanceEmbedding::embed(int dist) {
  unsigned base = (dist < 0 ? max_bin : 0);
  unsigned dist_std = 0;
  if (dist) {
    dist_std = static_cast<unsigned>(log(dist < 0 ? -dist : dist) / log(1.6f)) + 1;
  }
  if (dist_std > max_bin) {
    dist_std = max_bin;
  }
  return (trainable ?
          dynet::expr::lookup(*cg, p_e, dist_std + base) :
          dynet::expr::const_lookup(*cg, p_e, dist_std + base));
}

BinnedDurationEmbedding::BinnedDurationEmbedding(dynet::Model& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable)
  : LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins, { dim, 1 })),
  max_bin(n_bins - 1) {
  BOOST_ASSERT(n_bins > 0);
}

void BinnedDurationEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::expr::Expression BinnedDurationEmbedding::embed(unsigned dur) {
  if (dur) {
    dur = static_cast<unsigned>(log(dur) / log(1.6f)) + 1;
  }
  if (dur > max_bin) {
    dur = max_bin;
  }
  return (trainable ? 
          dynet::expr::lookup((*cg), p_e, dur) :
          dynet::expr::const_lookup((*cg), p_e, dur));
}

CNNLayer::CNNLayer(dynet::Model& m,
                   unsigned input_dim,
                   unsigned output_dim,
                   const std::vector<std::pair<unsigned, unsigned>>& info,
                   bool trainable) :
  LayerI(trainable),
  filters_info(info),
  input_dim(input_dim) {
  unsigned n_filter_types = info.size();
  unsigned combined_dim = 0;
  p_filters.resize(n_filter_types);
  p_biases.resize(n_filter_types);
  for (unsigned i = 0; i < info.size(); ++i) {
    const auto& filter_width = info[i].first;
    const auto& nb_filters = info[i].second;
    p_filters[i].resize(nb_filters);
    p_biases[i].resize(nb_filters);
    for (unsigned j = 0; j < nb_filters; ++j) {
      p_filters[i][j] = m.add_parameters({ input_dim, filter_width });
      p_biases[i][j] = m.add_parameters({ input_dim });
      combined_dim += input_dim;
    }
  }
  p_W = m.add_parameters({ output_dim, combined_dim });
}

void CNNLayer::new_graph(dynet::ComputationGraph& hg) {
  filters.resize(p_filters.size());
  for (unsigned i = 0; i < filters.size(); ++i) {
    filters[i].resize(p_filters[i].size());
    for (unsigned j = 0; j < filters[i].size(); ++j) {
      filters[i][j] = (trainable ?
                       dynet::expr::parameter(hg, p_filters[i][j]) :
                       dynet::expr::const_parameter(hg, p_filters[i][j]));
    }
  }
  biases.resize(p_biases.size());
  for (unsigned i = 0; i < biases.size(); ++i) {
    biases[i].resize(p_filters[i].size());
    for (unsigned j = 0; j < biases[i].size(); ++j) {
      biases[i][j] = (trainable ?
                      dynet::expr::parameter(hg, p_biases[i][j]) :
                      dynet::expr::parameter(hg, p_biases[i][j]));
    }
  }
  W = (trainable ? dynet::expr::parameter(hg, p_W) : dynet::expr::const_parameter(hg, p_W));
  padding = dynet::expr::zeroes(hg, { input_dim });
}

dynet::expr::Expression CNNLayer::get_output(const std::vector<dynet::expr::Expression>& c) {
  std::vector<dynet::expr::Expression> s(c);
  std::vector<dynet::expr::Expression> tmp;
  for (unsigned ii = 0; ii < filters_info.size(); ++ii) {
    const auto& filter_width = filters_info[ii].first;
    const auto& nb_filters = filters_info[ii].second;

    for (unsigned p = 0; p < filter_width - 1; ++p) { s.push_back(padding); }
    for (unsigned jj = 0; jj < nb_filters; ++jj) {
      auto& filter = filters[ii][jj];
      auto& bias = biases[ii][jj];
      auto t = dynet::expr::conv1d_narrow(dynet::expr::concatenate_cols(s), filter);
      t = colwise_add(t, bias);
      t = dynet::expr::rectify(dynet::expr::kmax_pooling(t, 1));
      tmp.push_back(t);
    }
    for (unsigned p = 0; p < filter_width - 1; ++p) { s.pop_back(); }
  }
  return W * dynet::expr::concatenate(tmp);
}

SoftmaxLayer::SoftmaxLayer(dynet::Model& m,
                           unsigned dim_input,
                           unsigned dim_output,
                           bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W(m.add_parameters({ dim_output, dim_input })) {
}

void SoftmaxLayer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W = dynet::expr::parameter(hg, p_W);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W = dynet::expr::const_parameter(hg, p_W);
  }
}

dynet::expr::Expression SoftmaxLayer::get_output(const dynet::expr::Expression& expr) {
  return dynet::expr::log_softmax(dynet::expr::affine_transform({B, W, expr}));
}

DenseLayer::DenseLayer(dynet::Model& m,
                       unsigned dim_input,
                       unsigned dim_output,
                       bool trainable) :
  LayerI(trainable),
  p_W(m.add_parameters({ dim_output, dim_input })),
  p_B(m.add_parameters({ dim_output, 1 })) {
}

void DenseLayer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    W = dynet::expr::parameter(hg, p_W);
    B = dynet::expr::parameter(hg, p_B);
  } else {
    W = dynet::expr::const_parameter(hg, p_W);
    B = dynet::expr::const_parameter(hg, p_B);
  }
}

dynet::expr::Expression DenseLayer::get_output(const dynet::expr::Expression& expr) {
  return dynet::expr::affine_transform({ B, W, expr });
}

Merge2Layer::Merge2Layer(dynet::Model& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })) {
}

void Merge2Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W1 = dynet::expr::parameter(hg, p_W1);
    W2 = dynet::expr::parameter(hg, p_W2);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W1 = dynet::expr::const_parameter(hg, p_W1);
    W2 = dynet::expr::const_parameter(hg, p_W2);
  }
}

dynet::expr::Expression Merge2Layer::get_output(const dynet::expr::Expression& expr1,
                                                const dynet::expr::Expression& expr2) {
  return dynet::expr::affine_transform({B, W1, expr1, W2, expr2});
}

Merge3Layer::Merge3Layer(dynet::Model& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })) {
}

void Merge3Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W1 = dynet::expr::parameter(hg, p_W1);
    W2 = dynet::expr::parameter(hg, p_W2);
    W3 = dynet::expr::parameter(hg, p_W3);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W1 = dynet::expr::const_parameter(hg, p_W1);
    W2 = dynet::expr::const_parameter(hg, p_W2);
    W3 = dynet::expr::const_parameter(hg, p_W3);
  }
}

dynet::expr::Expression Merge3Layer::get_output(const dynet::expr::Expression& expr1,
                                                const dynet::expr::Expression& expr2,
                                                const dynet::expr::Expression& expr3) {
  return dynet::expr::affine_transform({B, W1, expr1, W2, expr2, W3, expr3});
}

Merge4Layer::Merge4Layer(dynet::Model& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })) {
}

void Merge4Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W1 = dynet::expr::parameter(hg, p_W1);
    W2 = dynet::expr::parameter(hg, p_W2);
    W3 = dynet::expr::parameter(hg, p_W3);
    W4 = dynet::expr::parameter(hg, p_W4);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W1 = dynet::expr::const_parameter(hg, p_W1);
    W2 = dynet::expr::const_parameter(hg, p_W2);
    W3 = dynet::expr::const_parameter(hg, p_W3);
    W4 = dynet::expr::const_parameter(hg, p_W4);
  }
}

dynet::expr::Expression Merge4Layer::get_output(const dynet::expr::Expression& expr1,
                                                const dynet::expr::Expression& expr2,
                                                const dynet::expr::Expression& expr3,
                                                const dynet::expr::Expression& expr4) {
  return dynet::expr::affine_transform({B, W1, expr1, W2, expr2, W3, expr3, W4, expr4});
}

Merge5Layer::Merge5Layer(dynet::Model& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })) {
}

void Merge5Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W1 = dynet::expr::parameter(hg, p_W1);
    W2 = dynet::expr::parameter(hg, p_W2);
    W3 = dynet::expr::parameter(hg, p_W3);
    W4 = dynet::expr::parameter(hg, p_W4);
    W5 = dynet::expr::parameter(hg, p_W5);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W1 = dynet::expr::const_parameter(hg, p_W1);
    W2 = dynet::expr::const_parameter(hg, p_W2);
    W3 = dynet::expr::const_parameter(hg, p_W3);
    W4 = dynet::expr::const_parameter(hg, p_W4);
    W5 = dynet::expr::const_parameter(hg, p_W5);
  }
}

dynet::expr::Expression Merge5Layer::get_output(const dynet::expr::Expression& expr1,
                                                const dynet::expr::Expression& expr2,
                                                const dynet::expr::Expression& expr3,
                                                const dynet::expr::Expression& expr4,
                                                const dynet::expr::Expression& expr5) {
  return dynet::expr::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5
  });
}

Merge6Layer::Merge6Layer(dynet::Model& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_input6,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 })),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })), 
  p_W6(m.add_parameters({ dim_output, dim_input6 })) {
}

void Merge6Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::expr::parameter(hg, p_B);
    W1 = dynet::expr::parameter(hg, p_W1);
    W2 = dynet::expr::parameter(hg, p_W2);
    W3 = dynet::expr::parameter(hg, p_W3);
    W4 = dynet::expr::parameter(hg, p_W4);
    W5 = dynet::expr::parameter(hg, p_W5);
    W6 = dynet::expr::parameter(hg, p_W6);
  } else {
    B = dynet::expr::const_parameter(hg, p_B);
    W1 = dynet::expr::const_parameter(hg, p_W1);
    W2 = dynet::expr::const_parameter(hg, p_W2);
    W3 = dynet::expr::const_parameter(hg, p_W3);
    W4 = dynet::expr::const_parameter(hg, p_W4);
    W5 = dynet::expr::const_parameter(hg, p_W5);
    W6 = dynet::expr::const_parameter(hg, p_W6);
  }
}

dynet::expr::Expression Merge6Layer::get_output(
  const dynet::expr::Expression& expr1,
  const dynet::expr::Expression& expr2,
  const dynet::expr::Expression& expr3,
  const dynet::expr::Expression& expr4,
  const dynet::expr::Expression& expr5,
  const dynet::expr::Expression& expr6) {
  return dynet::expr::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5, W6, expr6
  });
}

SegUniEmbedding::SegUniEmbedding(dynet::Model& m, unsigned n_layers,
  unsigned lstm_input_dim, unsigned seg_dim)
  :
  p_h0(m.add_parameters({ lstm_input_dim })),
  builder(n_layers, lstm_input_dim, seg_dim, m) {
}

void SegUniEmbedding::construct_chart(dynet::ComputationGraph& cg,
                                      const std::vector<dynet::expr::Expression>& c,
                                      int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);
  dynet::expr::Expression h0 = dynet::expr::parameter(cg, p_h0);
  builder.new_graph(cg);
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    builder.start_new_sequence();
    builder.add_input(h0);
    // Put one span in h[i][j]
    for (unsigned k = 0; k < seg_len; ++k) {
      hi[k] = builder.add_input(c[i + k]);
    }
  }
}

const dynet::expr::Expression& SegUniEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j <= len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}

void SegUniEmbedding::set_dropout(float& rate) {
  builder.set_dropout(rate);
}

void SegUniEmbedding::disable_dropout() {
  builder.disable_dropout();
}

SegBiEmbedding::SegBiEmbedding(dynet::Model& m, unsigned n_layers,
  unsigned lstm_input_dim, unsigned seg_dim)
  :
  fwd(m, n_layers, lstm_input_dim, seg_dim),
  bwd(m, n_layers, lstm_input_dim, seg_dim) {
}

void SegBiEmbedding::construct_chart(dynet::ComputationGraph& cg,
  const std::vector<dynet::expr::Expression>& c,
  int max_seg_len) {
  len = c.size();
  fwd.construct_chart(cg, c, max_seg_len);
  std::vector<dynet::expr::Expression> rc(len);
  for (unsigned i = 0; i < len; ++i) { rc[i] = c[len - i - 1]; }
  bwd.construct_chart(cg, rc, max_seg_len);
  h.clear();
  h.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    auto& hi = h[i];
    unsigned seg_len = max_j - i;
    hi.resize(seg_len);
    for (unsigned k = 0; k < seg_len; ++k) {
      unsigned j = i + k;
      const dynet::expr::Expression& fe = fwd(i, j);
      const dynet::expr::Expression& be = bwd(len - 1 - j, len - 1 - i);
      hi[k] = std::make_pair(fe, be);
    }
  }
}

const SegBiEmbedding::ExpressionPair& SegBiEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j <= len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}

void SegBiEmbedding::set_dropout(float& rate) {
  fwd.set_dropout(rate);
  bwd.set_dropout(rate);
}

void SegBiEmbedding::disable_dropout() {
  fwd.disable_dropout();
  bwd.disable_dropout();
}

SegConcateEmbedding::SegConcateEmbedding(dynet::Model& m,
                                         unsigned input_dim,
                                         unsigned output_dim,
                                         unsigned max_seg_len)
  : p_W(m.add_parameters({ output_dim, input_dim * max_seg_len })),
  p_b(m.add_parameters({ output_dim })),
  paddings(input_dim, 0.) {
}

void SegConcateEmbedding::construct_chart(dynet::ComputationGraph& cg,
                                          const std::vector<dynet::expr::Expression>& c,
                                          int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);

  auto W = dynet::expr::parameter(cg, p_W);
  auto b = dynet::expr::parameter(cg, p_b);
  auto dim = p_W.dim().cols() / max_seg_len;
  auto z = dynet::expr::zeroes(cg, { dim });
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    std::vector<dynet::expr::Expression> s(max_seg_len, z);
    for (unsigned k = 0; k < seg_len; ++k) {
      s[k] = c[i + k];
      hi[k] = dynet::expr::rectify(dynet::expr::affine_transform({ b, W, dynet::expr::concatenate(s) }));
    }
  }
}

const dynet::expr::Expression& SegConcateEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}
