#include "corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

const char* Corpus::UNK  = "_UNK_";
const char* Corpus::BAD0 = "_BAD0_";
const char* Corpus::ROOT = "_ROOT_";
const unsigned Corpus::BAD_HED = 10000;
const unsigned Corpus::BAD_DEL = 10000;
const unsigned Corpus::UNDEF_HED = 20000;
const unsigned Corpus::UNDEF_DEL = 20000;

void parse_to_vector(const ParseUnits& parse,
                     std::vector<unsigned>& heads,
                     std::vector<unsigned>& deprels) {
  heads.clear();
  deprels.clear();
  for (unsigned i = 0; i < parse.size(); ++i) {
    if (parse[i].head < Corpus::BAD_HED) {
      heads.push_back(parse[i].head - 1);
    } else {
      heads.push_back(parse[i].head);
    }
    deprels.push_back(parse[i].deprel);
  }
}

void vector_to_parse(const std::vector<unsigned>& heads,
                     const std::vector<unsigned>& deprels,
                     ParseUnits& parse) {
  parse.clear();
  BOOST_ASSERT_MSG(heads.size() == deprels.size(),
    "In corpus.cc: vector_to_parse, #heads should be equal to #deprels");

  for (unsigned i = 0; i < heads.size(); ++i) {
    ParseUnit parse_unit;
    parse_unit.head = (heads[i] < Corpus::BAD_HED ? heads[i] + 1 : heads[i]);
    parse_unit.deprel = deprels[i];
    parse.push_back(parse_unit);
  }
}

Corpus::Corpus() : n_train(0), n_devel(0) {

}

void Corpus::load_training_data(const std::string& filename,
                                bool allow_partial_tree) {
  _INFO << "Corpus:: reading training data from: " << filename;

  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);
  word_map.insert(Corpus::ROOT);
  char_map.insert(Corpus::BAD0);
  char_map.insert(Corpus::UNK);
  char_map.insert(Corpus::ROOT);
  pos_map.insert(Corpus::ROOT);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, training_inputs[n_train], training_parses[n_train], true, allow_partial_tree);
      data = "";
      ++n_train;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, training_inputs[n_train], training_parses[n_train], true, allow_partial_tree);
    ++n_train;
  }

  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
}

void Corpus::load_devel_data(const std::string& filename,
                             bool allow_new_postag_and_deprels) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
    "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      parse_data(data, devel_inputs[n_devel], devel_parses[n_devel], false, allow_new_postag_and_deprels);
      data = "";
      ++n_devel;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, devel_inputs[n_devel], devel_parses[n_devel], false, allow_new_postag_and_deprels);
    ++n_devel;
  }

  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
}

unsigned utf8_len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else abort();
}

// id form lemma cpos pos feat head deprel phead pdeprel
// 0  1    2     3    4   5    6     7     8     9
void Corpus::parse_data(const std::string& data,
                        InputUnits& input_units,
                        ParseUnits& parse_units,
                        bool train,
                        bool allow_partial_tree) {
  std::stringstream S(data);
  std::string line;

  input_units.clear();
  parse_units.clear();

  InputUnit input_unit;
  ParseUnit parse_unit;

  while (std::getline(S, line)) {
    std::vector<std::string> tokens;
    boost::algorithm::trim(line);
    boost::algorithm::split(tokens, line, boost::is_any_of("\t"), boost::token_compress_on);

    if (boost::algorithm::to_lower_copy(tokens[1]) == "-lrb-") { tokens[1] = "("; }
    else if (boost::algorithm::to_lower_copy(tokens[1]) == "-rrb-") { tokens[1] = ")"; }

    BOOST_ASSERT_MSG(tokens.size() > 6, "Corpus:: Illegal conll format!");
    
    if (train) {
      input_unit.wid = word_map.insert(tokens[1]);
      input_unit.nid = (norm_map.contains(tokens[2]) ? norm_map.get(tokens[2]) : norm_map.get(Corpus::UNK));
      input_unit.pid = pos_map.insert(tokens[3]);

      unsigned cur = 0;
      while (cur < tokens[1].size()) {
        unsigned len = utf8_len(tokens[1][cur]);
        input_unit.cids.push_back(char_map.insert(tokens[1].substr(cur, len)));
        cur += len;
      }

      input_unit.aux_wid = input_unit.wid;
      input_unit.w_str = tokens[1];
      input_unit.n_str = tokens[2];
      input_unit.f_str = tokens[5];

      if (allow_partial_tree && tokens[6] == "_") {
        parse_unit.head = UNDEF_HED;
        parse_unit.deprel = UNDEF_DEL;
      } else {
        parse_unit.head = boost::lexical_cast<unsigned>(tokens[6]);
        parse_unit.deprel = deprel_map.insert(tokens[7]);
      }
      input_units.push_back(input_unit);
      parse_units.push_back(parse_unit);
    } else {
      input_unit.wid = (word_map.contains(tokens[1]) ? word_map.get(tokens[1]) : word_map.get(UNK));
      input_unit.nid = (norm_map.contains(tokens[2]) ? norm_map.get(tokens[2]) : norm_map.get(UNK));
      input_unit.pid = pos_map.get(tokens[3]);

      unsigned cur = 0;
      while (cur < tokens[1].size()) {
        unsigned len = utf8_len(tokens[1][cur]);
        std::string ch_str = tokens[1].substr(cur, len);
        input_unit.cids.push_back(
          char_map.contains(ch_str) ? char_map.get(ch_str) : char_map.get(Corpus::UNK)
        );
        cur += len;
      }

      input_unit.aux_wid = input_unit.wid;
      input_unit.w_str = tokens[1];
      input_unit.n_str = tokens[2];
      input_unit.f_str = tokens[5];
      
      parse_unit.head = boost::lexical_cast<unsigned>(tokens[6]);
      if (allow_partial_tree) {
        // Partial tree not allowed in development data, if allow partial tree set, meaning new label
        // label is allow.
        parse_unit.deprel = deprel_map.insert(tokens[7]);
      } else {
        parse_unit.deprel = deprel_map.get(tokens[7]);
      }
      input_units.push_back(input_unit);
      parse_units.push_back(parse_unit);
    }
  }
  input_unit.wid = word_map.get(ROOT);
  input_unit.nid = norm_map.get(ROOT);
  input_unit.pid = pos_map.get(ROOT);
  input_unit.aux_wid = input_unit.wid;
  input_unit.w_str = ROOT;
  input_unit.n_str = ROOT;
  input_unit.f_str = ROOT;
  input_units.push_back(input_unit);
  
  parse_unit.head = BAD_HED;
  parse_unit.deprel = BAD_DEL;
  parse_units.push_back(parse_unit);

  unsigned dummy_root = input_units.size();
  // reset root to the DUMMY ROOT
  for (auto& parse_unit : parse_units) {
    if (parse_unit.head == 0) {
      parse_unit.head = dummy_root;
    }
  }
}

unsigned Corpus::get_or_add_word(const std::string& word) {
  return word_map.insert(word);
}

void Corpus::stat() {
  _INFO << "Corpus:: # of words = " << word_map.size();
  _INFO << "Corpus:: # of norm = " << norm_map.size();
  _INFO << "Corpus:: # of char = " << char_map.size();
  _INFO << "Corpus:: # of pos = " << pos_map.size();
  _INFO << "Corpus:: # of deprel = " << deprel_map.size();
}

void Corpus::load_word_embeddings(const std::string & embedding_file,
                                  unsigned pretrained_dim,
                                  Embeddings & pretrained) {
  pretrained[norm_map.insert(Corpus::BAD0)] = std::vector<float>(pretrained_dim, 0.);
  pretrained[norm_map.insert(Corpus::UNK)] = std::vector<float>(pretrained_dim, 0.);
  pretrained[norm_map.insert(Corpus::ROOT)] = std::vector<float>(pretrained_dim, 0.);
  _INFO << "Corpus:: Loading from " << embedding_file << " with " << pretrained_dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  // get the header in word2vec styled embedding.
  std::getline(ifs, line);
  std::vector<float> v(pretrained_dim, 0.);
  std::string word;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> word;
    // actually, there should be a checking about the embedding dimension.
    for (unsigned i = 0; i < pretrained_dim; ++i) { iss >> v[i]; }
    unsigned id = norm_map.insert(word);
    pretrained[id] = v;
  }
  _INFO << "Corpus:: loaded " << pretrained.size() << " entries.";
}

void Corpus::load_empty_embeddings(unsigned pretrained_dim,
                                   Embeddings & pretrained) {
  pretrained[norm_map.insert(Corpus::BAD0)] = std::vector<float>(pretrained_dim, 0.);
  pretrained[norm_map.insert(Corpus::UNK)] = std::vector<float>(pretrained_dim, 0.);
  pretrained[norm_map.insert(Corpus::ROOT)] = std::vector<float>(pretrained_dim, 0.);
  _INFO << "Corpus:: loaded " << pretrained.size() << " entries.";
}

void Corpus::get_vocabulary_and_word_count() {
  for (auto& payload : training_inputs) {
    for (auto& item : payload.second) {
      training_vocab.insert(item.wid);
      ++counter[item.wid];
    }
  }
}
