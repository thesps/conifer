#ifndef CONIFER_CPP_QS_H__
#define CONIFER_CPP_QS_H__
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 QuickScorer tree ensemble traversal. Implementation of Algorithm 2 from
 https://doi.org/10.1145/2766462.2767733 in Python and Numpy.

 Rather than traversing each tree root-to-leaf, QuickScorer represents every
 internal node with a bitvector that masks off the leaves of its left subtree,
 and performs an interleaved, feature-by-feature traversal of the whole
 ensemble.

   Step 1: for every feature k, the thresholds of all the nodes testing it
     (across all trees of the ensemble) are stored in a sorted array. Given
     an input x, the nodes whose test fails form a prefix of that (sorted)
 array, delimited by the position of x[k] among the sorted thresholds. The
 bitvector of each false node is ANDed into the result bitvector v[h] of the
 tree h the node belongs to.

   Step 2: after all features are processed, the exit leaf of tree h is the
     leftmost set bit in v[h] (Theorem 1 of the paper). The output values
     of the exit leaves are summed to produce the score.

  BWQS (blockwise QS) improves the cache behaviour of QS on large ensembles:
  the ensemble is split into disjoint blocks of 'tau' trees, each with its own
  copy of the relevant data structures, and blocks of 'delta' samples are scored
  together over one tree block before moving to the next.

 Differences from the paper's design:
  - Similar to the Python QS backend, the bit order is mirrored relative
    to the paper.
  - Bitvectors are always full 64-bit words, so trees may have at most 64
    leaves. The paper instead pads bitvectors to B in {1, 2, 4, 8} bytes
    fitting the widest tree of the ensemble, saving on space.
  - Thresholds, leaf values and inputs are held in the conifer emulation
    types (e.g. ap_fixed) rather than the paper's float, so that QS scores
    are identical to the cpp backend for the same precision config.
 */

namespace conifer_qs {

class ConiferConfiguration {
public:
  using score_t = float;
  using threshold_t = float;
  using input_t = float;
  using weight_t = float;
};

/* Tree as stored in the conifer project JSON */
class RawTree {
public:
  std::vector<int> feature;
  std::vector<std::vector<double>> weight;
  std::vector<int> children_left;
  std::vector<int> children_right;
  std::vector<double> threshold;
  std::vector<double> value;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RawTree, feature, weight, children_left,
                                 children_right, threshold, value);
};

/* n low bits set, valid for n in [0, 64] */
inline uint64_t ones_mask(size_t n) {
  return n >= 64 ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
}

/*
 * The QuickScorer data structures and interleaved traversal for one block of
 * trees.
 */
template <typename Config> class TreeBlock {
public:
  using T = typename Config::threshold_t;
  using U = typename Config::score_t;

  size_t n_trees = 0;    // trees in block
  size_t max_leaves = 0; // widest tree of block
  // per-node data, grouped per feature with ascending thresholds within each
  // group
  std::vector<T> thresholds;
  std::vector<uint32_t> tree_ids;
  std::vector<uint64_t> bitvectors;
  std::vector<size_t> offsets;  // start of each feature's group, n_features + 1
  std::vector<uint64_t> init_v; // per-tree initial result bitvector
  std::vector<U> leaves; // leaf values, [n_trees][max_leaves], left-to-right

  TreeBlock(const std::vector<const RawTree *> &block_trees,
            size_t n_features) {
    n_trees = block_trees.size();
    struct Node {
      T threshold;
      uint32_t tree;
      uint64_t bitvector;
    };
    std::vector<std::vector<Node>> per_feature(n_features);
    std::vector<std::vector<U>> leaf_values(n_trees);

    for (uint32_t h = 0; h < n_trees; h++) {
      const RawTree &t = *block_trees.at(h);
      std::vector<int> leaf_nodes;
      std::vector<std::array<int, 3>> left_ranges;
      leaf_layout(t, 0, leaf_nodes, left_ranges);
      size_t n_leaves = leaf_nodes.size();
      if (n_leaves > 64) {
        throw std::runtime_error(
            "QuickScorer bitvectors are 64-bit words, but a tree has " +
            std::to_string(n_leaves) + " leaves");
      }
      uint64_t ones = ones_mask(n_leaves);
      for (const auto &r : left_ranges) {
        int n = r[0], a = r[1], b = r[2];
        check_axis_aligned(t, n);
        // construct node bitvector: 0s at the leaves of the left subtree (bits
        // a..b), 1s elsewhere
        uint64_t bv = ones ^ (ones_mask(b - a + 1) << a);
        per_feature.at(t.feature.at(n))
            .push_back({(T)t.threshold.at(n), h, bv});
      }
      leaf_values.at(h).reserve(n_leaves);
      for (int n : leaf_nodes)
        leaf_values.at(h).push_back((U)t.value.at(n));
      init_v.push_back(ones);
      max_leaves = std::max(max_leaves, n_leaves);
    }

    // concatenate the per-feature groups & sort in the quantized threshold
    // domain, since that is what the traversal compares in
    offsets.assign(n_features + 1, 0);
    for (size_t k = 0; k < n_features; k++) {
      auto &nodes = per_feature.at(k);
      std::stable_sort(nodes.begin(), nodes.end(),
                       [](const Node &a, const Node &b) {
                         return a.threshold < b.threshold;
                       });
      for (const auto &nd : nodes) {
        thresholds.push_back(nd.threshold);
        tree_ids.push_back(nd.tree);
        bitvectors.push_back(nd.bitvector);
      }
      offsets.at(k + 1) = thresholds.size();
    }

    // leaf value table padded to the widest tree
    leaves.assign(n_trees * max_leaves, U(0));
    for (size_t h = 0; h < n_trees; h++) {
      for (size_t j = 0; j < leaf_values.at(h).size(); j++) {
        leaves.at(h * max_leaves + j) = leaf_values.at(h).at(j);
      }
    }
  }

  /*
   * QS traversal of this block for one sample x.
   * 'strict' selects the splitting convention: if strict = false, a node is
   * false iff threshold < x ("<=" convention) or if strict = true, threshold <=
   * x ("<"). v is array of n_trees words and is provided by caller. the exit
   * leaf value of each tree is written to out.
   */
  void score(const T *x, bool strict, uint64_t *v, U *out) const {
    std::copy(init_v.begin(), init_v.end(), v);
    // Step 1: within a feature group thresholds ascend, so the false nodes
    // form a prefix => scan it linearly and stop at the first true node
    size_t n_features = offsets.size() - 1;
    for (size_t k = 0; k < n_features; k++) {
      T xk = x[k];
      for (size_t i = offsets[k]; i < offsets[k + 1]; i++) {
        bool false_node = strict ? (thresholds[i] <= xk) : (thresholds[i] < xk);
        if (!false_node)
          break;
        v[tree_ids[i]] &= bitvectors[i];
      }
    }
    // Step 2: the exit leaf is the lowest set bit of v (leaves numbered from
    // the LSB here). v is never 0 since the exit leaf bit is never cleared.
    for (size_t h = 0; h < n_trees; h++) {
      out[h] = leaves[h * max_leaves + __builtin_ctzll(v[h])];
    }
  }

  /* Total size in bytes of the traversal data structures */
  size_t nbytes() const {
    return thresholds.size() * sizeof(T) + tree_ids.size() * sizeof(uint32_t) +
           bitvectors.size() * sizeof(uint64_t) +
           offsets.size() * sizeof(size_t) + init_v.size() * sizeof(uint64_t) +
           leaves.size() * sizeof(U);
  }

private:
  /*
   * Number the leaves of the subtree at node n left-to-right and, for every
   * internal node, record the (contiguous, inclusive) range of leaf numbers in
   * its left subtree. Returns the leaf range of the subtree at n.
   */
  static std::pair<int, int>
  leaf_layout(const RawTree &t, int n, std::vector<int> &leaf_nodes,
              std::vector<std::array<int, 3>> &left_ranges) {
    if (t.feature.at(n) == -2) { // leaf
      int j = (int)leaf_nodes.size();
      leaf_nodes.push_back(n);
      return {j, j};
    }
    auto left = leaf_layout(t, t.children_left.at(n), leaf_nodes, left_ranges);
    auto right =
        leaf_layout(t, t.children_right.at(n), leaf_nodes, left_ranges);
    left_ranges.push_back({n, left.first, left.second});
    return {left.first, right.second};
  }

  /* Verify internal node n tests exactly one feature with unit weight */
  static void check_axis_aligned(const RawTree &t, int n) {
    const auto &node_weight_vec = t.weight.at(n);
    int node_feature_idx = t.feature.at(n);
    bool ok = node_feature_idx >= 0 &&
              (size_t)node_feature_idx < node_weight_vec.size() &&
              node_weight_vec.at(node_feature_idx) == 1.0;
    // check tgat every entry other than f == 0
    for (size_t i = 0; ok && i < node_weight_vec.size(); i++) {
      ok = (int)i == node_feature_idx || node_weight_vec.at(i) == 0.0;
    }
    if (!ok) {
      throw std::runtime_error("QuickScorer supports only axis-aligned trees "
                               "(one feature per split), "
                               "but node " +
                               std::to_string(n) +
                               " has a non one-hot weight vector");
    }
  }
}; // class TreeBlock

template <typename Config> class BDT {
public:
  using T = typename Config::threshold_t;
  using U = typename Config::score_t;

  unsigned int n_classes;
  unsigned int n_trees;
  unsigned int n_features;
  double norm;
  std::vector<double> init_predict;
  std::vector<U> init_predict_;
  // vector of DTs: trees[i_tree][i_class]
  std::vector<std::vector<RawTree>> trees;

  bool strict;   // splitting convention is "<" (rather than "<=")
  size_t n_flat; // total trees x classes pairs,
                 // index h = i_tree * n_classes + i_class
  size_t delta;  // documents per block, 0 = all documents in one block
  std::vector<TreeBlock<Config>>
      blocks;                        // blocks of tau trees, in flat tree order
  std::vector<size_t> block_offsets; // first flat tree index of each block

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(BDT, n_classes, n_trees, n_features, norm,
                                 init_predict, trees);

  /*
   * Construct the QuickScorer structures from a conifer project JSON file.
   * tau: trees per block, 0 (default) puts the whole ensemble in one block
   * delta: documents per block, 0 (default) scores all documents in one block
   */
  BDT(const std::string &filename, int tau = 0, int delta = 0) {
    std::ifstream ifs(filename);
    nlohmann::json j = nlohmann::json::parse(ifs);
    from_json(j, *this);
    // read the splitting convention with default value of "<=" if it's
    // unspecified
    std::string splitting_convention = j.value("splitting_convention", "<=");
    if (splitting_convention != "<" && splitting_convention != "<=") {
      throw std::invalid_argument("Invalid operator string: " +
                                  splitting_convention);
    }
    strict = splitting_convention == "<";
    if (n_classes == 2)
      n_classes = 1;
    for (double ip : init_predict)
      init_predict_.push_back((U)ip);

    // one bitvector lane per (tree, class) pair
    std::vector<const RawTree *> flat;
    for (const auto &tree_v : trees) {
      for (const auto &t : tree_v)
        flat.push_back(&t);
    }
    n_flat = flat.size();
    size_t tau_ = tau > 0 ? (size_t)tau : n_flat;
    this->delta = delta > 0 ? (size_t)delta : 0;
    for (size_t h0 = 0; h0 < n_flat; h0 += tau_) {
      size_t h1 = std::min(h0 + tau_, n_flat);
      blocks.emplace_back(
          std::vector<const RawTree *>(flat.begin() + h0, flat.begin() + h1),
          n_features);
      block_offsets.push_back(h0);
    }
  }

  /*
   * Score a batch of samples. BWQS loop structure, as in the paper: outer loop
   * over the tree blocks, then over blocks of delta documents, then over the
   * documents of the block, with a running partial score per document. Keeping
   * the tree blocks outermost means each tree block's (large, read-only) QS
   * structures are loaded once and reused for the whole scoring set while
   * resident in cache. X: row-major n_samples x n_features, returns row-major
   * n_samples x n_classes.
   */
  std::vector<double> decision_function_batch(const double *X, size_t n_samples,
                                              size_t n_features_in) const {
    if (n_features_in != n_features) {
      throw std::runtime_error("Wrong number of features, expected " +
                               std::to_string(n_features) + ", got " +
                               std::to_string(n_features_in));
    }
    // cast the inputs 
    std::vector<T> XT(n_samples * n_features);
    for (size_t i = 0; i < XT.size(); i++)
      XT[i] = (T)X[i];

    // running scores, starting from init_predict like the cpp backend's
    // accumulate
    std::vector<U> scores(n_samples * n_classes);
    for (size_t d = 0; d < n_samples; d++) {
      for (size_t c = 0; c < n_classes; c++)
        scores[d * n_classes + c] = init_predict_.at(c);
    }

    size_t max_block = 0;
    for (const auto &block : blocks)
      max_block = std::max(max_block, block.n_trees);
    std::vector<uint64_t> v(max_block);  // result bitvectors
    std::vector<U> leaf_vals(max_block); // exit leaf values

    size_t delta_ = delta > 0 ? delta : n_samples;
    // loop over the tree blocks
    for (size_t b = 0; b < blocks.size(); b++) {
      const TreeBlock<Config> &block = blocks[b];
      size_t h0 = block_offsets[b];
      // loop over blocks of delta documents
      for (size_t d0 = 0; d0 < n_samples; d0 += delta_) {
        size_t d1 = std::min(d0 + delta_, n_samples);
        for (size_t d = d0; d < d1; d++) {
          block.score(&XT[d * n_features], strict, v.data(), leaf_vals.data());
          // blocks are in flat tree order, so per class the additions happen in
          // tree order: same operation order as the cpp backend. Each document
          // visits the tree blocks in ascending order, so the ordering (and
          // hence the score) is unchanged by the block shape

          // to index by class: sd[c] == scores[d * n_classes + c]
          U *sd = &scores[d * n_classes];

          // add each tree's exit leaf value into appropriate class
          for (size_t hh = 0; hh < block.n_trees; hh++) {
            sd[(h0 + hh) % n_classes] += leaf_vals[hh];
          }
        }
      }
    }

    std::vector<double> y(n_samples * n_classes);
    for (size_t i = 0; i < y.size(); i++) {
      U yi = scores[i];
      yi *= (U)norm; // ensemble weighting in hardware precision
      y[i] = (double)yi;
    }
    return y;
  }

  /* Total size in bytes of the traversal data structures */
  size_t nbytes() const {
    size_t n = 0;
    for (const auto &block : blocks)
      n += block.nbytes();
    return n;
  }

  unsigned int get_n_classes() const { return n_classes; }

}; // class BDT

} // namespace conifer_qs

#endif
