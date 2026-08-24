#ifndef CONIFER_CPP_H__
#define CONIFER_CPP_H__
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace conifer{

/* ---
* Balanced tree reduce implementation.
* Reduces an array of inputs to a single value using the template binary operator 'Op',
* for example summing all elements with Op_add, or finding the maximum with Op_max
* Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
* before applying and accumulate the result over the rolled dimension.
* Required for emulation to guarantee equality of ordering.
* --- */
constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

template <int B>
constexpr int pow(int x) {
  return x == 0 ? 1 : B * pow<B>(x - 1);
}

constexpr int pow2(int x) { return pow<2>(x); }

template <class T, class Op>
T reduce(std::vector<T> x, Op op) {
  int N = x.size();
  int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
  //static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
  if (N == 1) {
    return x.at(0);
  } else if (N == 2) {
    return op(x.at(0), x.at(1));
  } else {
    std::vector<T> left(x.begin(), x.begin() + leftN);
    std::vector<T> right(x.begin() + leftN, x.end());
    return op(reduce<T, Op>(left, op), reduce<T, Op>(right, op));
  }
}

template<class T>
class OpAdd {
public:
  T operator()(T a, T b) { return a + b; }
};

template <typename T>
std::function<bool (T, T)> createSplit(const std::string& op) {
    std::function<bool (T, T)> split;
    if (op == "<") {
        split = [](const T& a, const T& b) { return a < b; };
    } else if (op == "<=") {
        split = [](const T& a, const T& b) { return a <= b; };
    } else {
        throw std::invalid_argument("Invalid operator string: " + op);
    }
    return split;
}

class ConiferConfiguration {
public:
  using score_t = float;
  using threshold_t = float;
  using weight_t = float;
  static const bool useAddTree = false;
};

template<typename Config> class TreeBlock;

template<typename Config>
class DecisionTree{
    static_assert(std::is_base_of<ConiferConfiguration, Config>::value,
                  "Config must derive from ConiferConfiguration");
private:
  using U = typename Config::score_t;
  using T = typename Config::threshold_t;
  using W = typename Config::weight_t;

  std::vector<int> feature;
  std::vector<std::vector<W>> weight_;
  std::vector<std::vector<double>> weight;
  std::vector<int> children_left;
  std::vector<int> children_right;
  std::vector<T> threshold_;
  std::vector<U> value_;
  std::vector<double> threshold;
  std::vector<double> value;
  std::function<bool (T, T)> split;

  // the QuickScorer traversal reads the tree structure to build its own layout
  friend class TreeBlock<Config>;

public:

  U decision_function(const std::vector<T> &x) const{
    /* Do the prediction */
    int i = 0;
    bool comparison;
    T accumulation = 0;
    while(feature[i] != -2){ // continue until reaching leaf
      accumulation = 0;
      // Multiply input x by weight vector, axis aligned uses a one hot encoded weight
      // Oblique uses a variable weight per feature
      for(unsigned int i_feat = 0; i_feat < weight_[i].size(); i_feat++){
        accumulation += x[i_feat] * weight_[i][i_feat];
      }
      comparison = split(accumulation, threshold_[i]);
      i = comparison ? children_left[i] : children_right[i];
    }
    return value_[i];
  }

  void init_(std::function<bool (T, T)> split){
    /* Since T, U types may not be readable from the JSON, read them to double and the cast them here */
    this->split = split;
    std::transform(threshold.begin(), threshold.end(), std::back_inserter(threshold_),
                   [](double t) -> T { return (T) t; });
    std::transform(value.begin(), value.end(), std::back_inserter(value_),
                   [](double v) -> U { return (U) v; });
    // Load weight vector of arrays, multidimensional so no nice one liner?
    for (int i = 0; i < weight.size(); i++) {
      weight_.push_back(std::vector<W>());
        for (int j = 0; j < weight[i].size(); j++) {
            weight_[i].push_back((W) weight[i][j]);
        }
    }
  }

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DecisionTree, feature, weight, children_left, children_right, threshold, value);

}; // class DecisionTree

/* ---
* QuickScorer tree ensemble traversal, Algorithm 2 of
* https://doi.org/10.1145/2766462.2767733, as an alternative to the root-to-leaf
* 'treewalk' of DecisionTree::decision_function above.
*
* Rather than traversing each tree root-to-leaf, QuickScorer represents every
* internal node with a bitvector that masks off the leaves of its left subtree,
* and performs an interleaved, feature-by-feature traversal of the whole ensemble.
*
*   Step 1: for every feature k, the thresholds of all the nodes testing it
*     (across all trees of the ensemble) are stored in a sorted array. Given an
*     input x, the nodes whose test fails form a prefix of that sorted array,
*     delimited by the position of x[k] among the sorted thresholds. The bitvector
*     of each false node is ANDed into the result bitvector v[h] of the tree h the
*     node belongs to.
*   Step 2: after all features are processed, the exit leaf of tree h is the
*     leftmost set bit in v[h] (Theorem 1 of the paper). The output values of the
*     exit leaves are summed to produce the score.
*
* BWQS (block-wise QuickScorer) improves the cache behaviour on large ensembles:
* the ensemble is split into disjoint blocks of 'tau' trees, each with its own copy
* of the relevant data structures, and blocks of 'delta' samples are scored together
* over one tree block before moving to the next. See docs/quickscorer.md for what
* each of the two block sizes is worth here.
*
* Differences from the paper's design:
*  - As in the python backend, the bit order is mirrored relative to the paper.
*  - Bitvectors are always full 64 bit words, so trees may have at most 64 leaves.
*    The paper instead pads bitvectors to B in {1, 2, 4, 8} bytes fitting the widest
*    tree of the ensemble, saving on space.
*  - Thresholds, leaf values and inputs are held in the emulation types (e.g.
*    ap_fixed) rather than the paper's float, so that the QuickScorer scores are
*    bit-identical to the treewalk ones for the same precision configuration.
* --- */

/* n low bits set, valid for n in [0, 64] */
inline uint64_t ones_mask(size_t n) {
  return n >= 64 ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
}

template<typename Config>
class TreeBlock{
    static_assert(std::is_base_of<ConiferConfiguration, Config>::value,
                  "Config must derive from ConiferConfiguration");
public:
  using T = typename Config::threshold_t;
  using U = typename Config::score_t;

  size_t n_trees = 0;    // trees in block
  size_t max_leaves = 0; // widest tree of block
  // per-node data, grouped per feature with ascending thresholds within each group
  std::vector<T> thresholds;
  std::vector<uint32_t> tree_ids;
  std::vector<uint64_t> bitvectors;
  std::vector<size_t> offsets;  // start of each feature's group, n_features + 1
  std::vector<uint64_t> init_v; // per-tree initial result bitvector
  std::vector<U> leaves;        // leaf values, [n_trees][max_leaves], left-to-right

  TreeBlock(const std::vector<const DecisionTree<Config>*> &block_trees, size_t n_features){
    n_trees = block_trees.size();
    struct Node {
      T threshold;
      uint32_t tree;
      uint64_t bitvector;
    };
    std::vector<std::vector<Node>> per_feature(n_features);
    std::vector<std::vector<U>> leaf_values(n_trees);

    for(uint32_t h = 0; h < n_trees; h++){
      const DecisionTree<Config> &t = *block_trees.at(h);
      std::vector<int> leaf_nodes;
      std::vector<std::array<int, 3>> left_ranges;
      leaf_layout(t, 0, leaf_nodes, left_ranges);
      size_t n_leaves = leaf_nodes.size();
      if(n_leaves > 64){
        throw std::runtime_error("QuickScorer bitvectors are 64-bit words, but a tree has "
                                 + std::to_string(n_leaves) + " leaves");
      }
      uint64_t ones = ones_mask(n_leaves);
      for(const auto &r : left_ranges){
        int n = r[0], a = r[1], b = r[2];
        check_axis_aligned(t, n);
        // construct node bitvector: 0s at the leaves of the left subtree (bits a..b), 1s elsewhere
        uint64_t bv = ones ^ (ones_mask(b - a + 1) << a);
        per_feature.at(t.feature.at(n)).push_back({t.threshold_.at(n), h, bv});
      }
      leaf_values.at(h).reserve(n_leaves);
      for(int n : leaf_nodes)
        leaf_values.at(h).push_back(t.value_.at(n));
      init_v.push_back(ones);
      max_leaves = std::max(max_leaves, n_leaves);
    }

    // concatenate the per-feature groups & sort in the quantized threshold domain,
    // since that is what the traversal compares in
    offsets.assign(n_features + 1, 0);
    for(size_t k = 0; k < n_features; k++){
      auto &nodes = per_feature.at(k);
      std::stable_sort(nodes.begin(), nodes.end(),
                       [](const Node &a, const Node &b){ return a.threshold < b.threshold; });
      for(const auto &nd : nodes){
        thresholds.push_back(nd.threshold);
        tree_ids.push_back(nd.tree);
        bitvectors.push_back(nd.bitvector);
      }
      offsets.at(k + 1) = thresholds.size();
    }

    // leaf value table padded to the widest tree
    leaves.assign(n_trees * max_leaves, U(0));
    for(size_t h = 0; h < n_trees; h++){
      for(size_t j = 0; j < leaf_values.at(h).size(); j++){
        leaves.at(h * max_leaves + j) = leaf_values.at(h).at(j);
      }
    }
  }

  /*
   * QuickScorer traversal of this block for one sample x.
   * 'strict' selects the splitting convention: a node is false iff threshold < x
   * ("<=" convention) when strict is false, or threshold <= x ("<") when it is true.
   * v is an array of n_trees words provided by the caller, the exit leaf value of
   * each tree is written to out.
   */
  void score(const T *x, bool strict, uint64_t *v, U *out) const{
    std::copy(init_v.begin(), init_v.end(), v);
    // Step 1: within a feature group thresholds ascend, so the false nodes form a
    // prefix => scan it linearly and stop at the first true node
    size_t n_features = offsets.size() - 1;
    for(size_t k = 0; k < n_features; k++){
      T xk = x[k];
      for(size_t i = offsets[k]; i < offsets[k + 1]; i++){
        bool false_node = strict ? (thresholds[i] <= xk) : (thresholds[i] < xk);
        if(!false_node)
          break;
        v[tree_ids[i]] &= bitvectors[i];
      }
    }
    // Step 2: the exit leaf is the lowest set bit of v (leaves numbered from the LSB
    // here). v is never 0 since the exit leaf bit is never cleared.
    for(size_t h = 0; h < n_trees; h++){
      out[h] = leaves[h * max_leaves + __builtin_ctzll(v[h])];
    }
  }

  /* Total size in bytes of the traversal data structures */
  size_t nbytes() const{
    return thresholds.size() * sizeof(T) + tree_ids.size() * sizeof(uint32_t)
           + bitvectors.size() * sizeof(uint64_t) + offsets.size() * sizeof(size_t)
           + init_v.size() * sizeof(uint64_t) + leaves.size() * sizeof(U);
  }

private:
  /*
   * Number the leaves of the subtree at node n left-to-right and, for every internal
   * node, record the (contiguous, inclusive) range of leaf numbers in its left
   * subtree. Returns the leaf range of the subtree at n.
   */
  static std::pair<int, int> leaf_layout(const DecisionTree<Config> &t, int n,
                                         std::vector<int> &leaf_nodes,
                                         std::vector<std::array<int, 3>> &left_ranges){
    if(t.feature.at(n) == -2){ // leaf
      int j = (int) leaf_nodes.size();
      leaf_nodes.push_back(n);
      return {j, j};
    }
    auto left = leaf_layout(t, t.children_left.at(n), leaf_nodes, left_ranges);
    auto right = leaf_layout(t, t.children_right.at(n), leaf_nodes, left_ranges);
    left_ranges.push_back({n, left.first, left.second});
    return {left.first, right.second};
  }

  /* Verify internal node n tests exactly one feature with unit weight */
  static void check_axis_aligned(const DecisionTree<Config> &t, int n){
    const auto &node_weight_vec = t.weight.at(n);
    int node_feature_idx = t.feature.at(n);
    bool ok = node_feature_idx >= 0
              && (size_t) node_feature_idx < node_weight_vec.size()
              && node_weight_vec.at(node_feature_idx) == 1.0;
    // check that every entry other than the tested feature is 0
    for(size_t i = 0; ok && i < node_weight_vec.size(); i++){
      ok = (int) i == node_feature_idx || node_weight_vec.at(i) == 0.0;
    }
    if(!ok){
      throw std::runtime_error("QuickScorer supports only axis-aligned trees "
                               "(one feature per split), but node "
                               + std::to_string(n) + " has a non one-hot weight vector");
    }
  }

}; // class TreeBlock

template<typename Config>
class BDT{
    static_assert(std::is_base_of<ConiferConfiguration, Config>::value,
                  "Config must derive from ConiferConfiguration");
private:

  using U = typename Config::score_t;
  using T = typename Config::threshold_t;
  using W = typename Config::weight_t;

  unsigned int n_classes;
  unsigned int n_trees;
  unsigned int n_features;
  double norm;
  std::vector<double> init_predict;
  std::vector<U> init_predict_;
  // vector of decision trees: outer dimension tree, inner dimension class
  std::vector<std::vector<DecisionTree<Config>>> trees;
  OpAdd<U> add;

  // QuickScorer traversal, empty until init_quickscorer is called
  bool strict;                              // splitting convention is "<" rather than "<="
  size_t qs_delta = 0;                      // samples per block, 0 = all samples in one block
  std::vector<TreeBlock<Config>> blocks;    // blocks of tau trees, in flat tree order
  std::vector<size_t> block_offsets;        // first flat tree index of each block

public:

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(BDT, n_classes, n_trees, n_features, norm, init_predict, trees);

  BDT(std::string filename){
    /* Construct the BDT from conifer cpp backend JSON file */
    std::ifstream ifs(filename);
    nlohmann::json j = nlohmann::json::parse(ifs);
    from_json(j, *this);
    auto splitting_convention = j.value("splitting_convention", "<="); // read the splitting convention with default value of "<=" if it's unspecified
    auto split = createSplit<T>(splitting_convention);
    strict = splitting_convention == "<";
    /* Do some transformation to initialise things into the proper emulation T, U types */
    if(n_classes == 2) n_classes = 1;
    std::transform(init_predict.begin(), init_predict.end(), std::back_inserter(init_predict_),
                   [](double ip) -> U { return (U) ip; });
    for(unsigned int i = 0; i < n_trees; i++){
      for(unsigned int j = 0; j < n_classes; j++){
        trees.at(i).at(j).init_(split);
      }
    }
  }

  std::vector<U> decision_function(std::vector<T> x) const{
    /* Do the prediction */
    assert("Size of feature vector mismatches expected n_features" && x.size() == n_features);
    std::vector<U> values;
    std::vector<std::vector<U>> values_trees;
    values_trees.resize(n_classes);
    values.resize(n_classes, U(0));
    for(unsigned int i = 0; i < n_classes; i++){
      std::transform(trees.begin(), trees.end(), std::back_inserter(values_trees.at(i)),
                     [&i, &x](const auto &tree_v){ return tree_v.at(i).decision_function(x); });
      if(Config::useAddTree){
        values.at(i) = init_predict_.at(i);
        values.at(i) += reduce<U, OpAdd<U>>(values_trees.at(i), add);
      }else{
        values.at(i) = std::accumulate(values_trees.at(i).begin(), values_trees.at(i).end(), U(init_predict_.at(i)));
      }
      values.at(i) *= (U) norm;
    }

    return values;
  }

  std::vector<double> _decision_function_double(std::vector<double> x) const{
    /* Do the prediction with data in/out as double, cast to T, U before prediction */
    std::vector<T> xt;
    std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](double xi) -> T { return (T) xi; });
    std::vector<U> y = decision_function(xt);
    std::vector<double> yd;
    std::transform(y.begin(), y.end(), std::back_inserter(yd),
                [](U yi) -> double { return (double) yi; });
    return yd;
  }

  /*
   * Build the QuickScorer traversal structures, replacing any previously built ones.
   * tau: trees per block, 0 (default) puts the whole ensemble in one block (plain QS)
   * delta: samples per block, 0 (default) scores all samples in one block
   */
  void init_quickscorer(int tau = 0, int delta = 0){
    blocks.clear();
    block_offsets.clear();
    // one bitvector lane per (tree, class) pair, flat index h = i_tree * n_classes + i_class
    std::vector<const DecisionTree<Config>*> flat;
    for(const auto &tree_v : trees){
      for(const auto &t : tree_v)
        flat.push_back(&t);
    }
    size_t n_flat = flat.size();
    size_t tau_ = tau > 0 ? (size_t) tau : n_flat;
    qs_delta = delta > 0 ? (size_t) delta : 0;
    for(size_t h0 = 0; h0 < n_flat; h0 += tau_){
      size_t h1 = std::min(h0 + tau_, n_flat);
      blocks.emplace_back(std::vector<const DecisionTree<Config>*>(flat.begin() + h0, flat.begin() + h1),
                          n_features);
      block_offsets.push_back(h0);
    }
  }

  /*
   * Score a batch of samples with the QuickScorer traversal. BWQS loop structure, as
   * in the paper: outer loop over the tree blocks, then over blocks of delta samples,
   * then over the samples of the block, with a running partial score per sample.
   * Keeping the tree blocks outermost means each tree block's (large, read-only)
   * QuickScorer structures are loaded once and reused for the whole scoring set while
   * resident in cache.
   * X: row-major n_samples x n_features, returns row-major n_samples x n_classes.
   */
  std::vector<double> _decision_function_batch_double(const double *X, size_t n_samples,
                                                      size_t n_features_in) const{
    if(blocks.empty()){
      throw std::runtime_error("QuickScorer structures are not initialised, "
                               "call init_quickscorer first");
    }
    if(n_features_in != n_features){
      throw std::runtime_error("Wrong number of features, expected "
                               + std::to_string(n_features) + ", got "
                               + std::to_string(n_features_in));
    }
    // cast the inputs to the emulation type
    std::vector<T> XT(n_samples * n_features);
    for(size_t i = 0; i < XT.size(); i++)
      XT[i] = (T) X[i];

    // running scores, starting from init_predict like the treewalk accumulate
    std::vector<U> scores(n_samples * n_classes);
    for(size_t d = 0; d < n_samples; d++){
      for(size_t c = 0; c < n_classes; c++)
        scores[d * n_classes + c] = init_predict_.at(c);
    }

    size_t max_block = 0;
    for(const auto &block : blocks)
      max_block = std::max(max_block, block.n_trees);
    std::vector<uint64_t> v(max_block);  // result bitvectors
    std::vector<U> leaf_vals(max_block); // exit leaf values

    size_t delta_ = qs_delta > 0 ? qs_delta : n_samples;
    // loop over the tree blocks
    for(size_t b = 0; b < blocks.size(); b++){
      const TreeBlock<Config> &block = blocks[b];
      size_t h0 = block_offsets[b];
      // loop over blocks of delta samples
      for(size_t d0 = 0; d0 < n_samples; d0 += delta_){
        size_t d1 = std::min(d0 + delta_, n_samples);
        for(size_t d = d0; d < d1; d++){
          block.score(&XT[d * n_features], strict, v.data(), leaf_vals.data());
          // blocks are in flat tree order, so per class the additions happen in tree
          // order: the same operation order as the treewalk accumulate. Each sample
          // visits the tree blocks in ascending order, so the ordering (and hence the
          // score) is unchanged by the block shape
          U *sd = &scores[d * n_classes]; // to index by class: sd[c]
          // add each tree's exit leaf value into the appropriate class
          for(size_t hh = 0; hh < block.n_trees; hh++){
            sd[(h0 + hh) % n_classes] += leaf_vals[hh];
          }
        }
      }
    }

    std::vector<double> y(n_samples * n_classes);
    for(size_t i = 0; i < y.size(); i++){
      U yi = scores[i];
      yi *= (U) norm; // ensemble weighting in hardware precision
      y[i] = (double) yi;
    }
    return y;
  }

  /* Total size in bytes of the QuickScorer traversal data structures (cf. Table 1 of the paper) */
  size_t nbytes() const{
    size_t n = 0;
    for(const auto &block : blocks)
      n += block.nbytes();
    return n;
  }

  unsigned int get_n_classes() const { return n_classes; }

}; // class BDT

} // namespace conifer

#endif
