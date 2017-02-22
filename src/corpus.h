#ifndef CORPUS_H
#define CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "alphabet.h"

struct InputUnit {
  std::vector<unsigned> cids;
  unsigned wid;
  unsigned nid;
  unsigned pid;
  unsigned aux_wid;
  std::string w_str;
  std::string n_str;
  std::string f_str;
};

struct ParseUnit {
  unsigned head;
  unsigned deprel;
};

typedef std::vector<InputUnit> InputUnits;
typedef std::vector<ParseUnit> ParseUnits;

void parse_to_vector(const ParseUnits& parse,
                     std::vector<unsigned>& heads,
                     std::vector<unsigned>& deprels);

void vector_to_parse(const std::vector<unsigned>& heads,
                     const std::vector<unsigned>& deprels,
                     ParseUnits& parse);

unsigned utf8_len(unsigned char x);

struct Corpus {
  const static char* UNK;
  const static char* BAD0;
  const static char* ROOT;
  const static unsigned BAD_HED;
  const static unsigned BAD_DEL;
  const static unsigned UNDEF_HED;
  const static unsigned UNDEF_DEL;

  unsigned n_train;
  unsigned n_devel;

  Alphabet word_map;
  Alphabet norm_map;
  Alphabet char_map;
  Alphabet pos_map;
  Alphabet deprel_map;

  std::unordered_map<unsigned, InputUnits> training_inputs;
  std::unordered_map<unsigned, ParseUnits> training_parses;

  std::unordered_map<unsigned, InputUnits> devel_inputs;
  std::unordered_map<unsigned, ParseUnits> devel_parses;

  std::set<unsigned> training_vocab;
  std::unordered_map<unsigned, unsigned> counter;

  Corpus();

  void load_training_data(const std::string& filename,
                          bool allow_partial_tree);

  void load_devel_data(const std::string& filename,
                       bool allow_new_postag_and_deprels);

  void parse_data(const std::string& data,
                  InputUnits& input_units, 
                  ParseUnits& parse_units,
                  bool train,
                  bool allow_partial_tree);
 
  void get_vocabulary_and_word_count();

  unsigned get_or_add_word(const std::string& word);
  void stat();

  void load_word_embeddings(const std::string& embedding_file,
                            unsigned pretrained_dim,
                            std::unordered_map<unsigned, std::vector<float> >& pretrained);

  void load_empty_embeddings(unsigned pretrained_dim,
                             std::unordered_map<unsigned, std::vector<float> >& pretrained);
};

#endif  //  end for CORPUS_H