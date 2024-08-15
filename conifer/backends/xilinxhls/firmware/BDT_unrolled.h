#ifndef BDT_H__
#define BDT_H__

#include "ap_fixed.h"

namespace BDT{

/* ---
* Balanced tree reduce implementation.
* Reduces an array of inputs to a single value using the template binary operator 'Op',
* for example summing all elements with OpAdd, or finding the maximum with OpMax
* Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
* before applying and accumulate the result over the rolled dimension.
* Required for emulation to guarantee equality of ordering.
* --- */
constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }

template <class T, int N, class Op> T reduce(const T *x, Op op) {
  static constexpr int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
  static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
  if (N == 1) {
    return x[0];
  }
  if (N == 2) {
    return op(x[0], x[1]);
  }
  return op(reduce<T, leftN, Op>(x, op), reduce<T, rightN, Op>(x + leftN, op));
}

template <class T> class OpAdd {
  public:
    T operator()(T a, T b) { return a + b; }
};

// Number of trees given number of classes
constexpr int fn_classes(int n_classes){
  return n_classes == 2 ? 1 : n_classes;
}

template<int n_tree, int n_nodes, int n_leaves, class input_t, class score_t, class threshold_t>
struct Tree {
private:
public:
  int feature[n_nodes];
  threshold_t threshold[n_nodes];
  score_t value[n_nodes];
  int children_left[n_nodes];
  int children_right[n_nodes];
  int parent[n_nodes];

  score_t decision_function(input_t x) const{
    #pragma HLS pipeline II = 1
    #pragma HLS ARRAY_PARTITION variable=feature
    #pragma HLS ARRAY_PARTITION variable=threshold
    #pragma HLS ARRAY_PARTITION variable=value
    #pragma HLS ARRAY_PARTITION variable=children_left
    #pragma HLS ARRAY_PARTITION variable=children_right
    #pragma HLS ARRAY_PARTITION variable=parent
    // These resource pragmas prevent the array of trees from being partitioned
    // They should be unnecessary anyway due to their own partitioning above
    /*#pragma HLS RESOURCE variable=feature core=ROM_nP_LUTRAM
    #pragma HLS RESOURCE variable=threshold core=ROM_nP_LUTRAM
    #pragma HLS RESOURCE variable=value core=ROM_nP_LUTRAM
    #pragma HLS RESOURCE variable=children_left core=ROM_nP_LUTRAM
    #pragma HLS RESOURCE variable=children_right core=ROM_nP_LUTRAM
    #pragma HLS RESOURCE variable=parent core=ROM_nP_LUTRAM*/

    bool comparison[n_nodes];
    bool activation[n_nodes];
    bool activation_leaf[n_leaves];
    score_t value_leaf[n_leaves];

    #pragma HLS ARRAY_PARTITION variable=comparison
    #pragma HLS ARRAY_PARTITION variable=activation
    #pragma HLS ARRAY_PARTITION variable=activation_leaf
    #pragma HLS ARRAY_PARTITION variable=value_leaf

    // Execute all comparisons
    Compare: for(int i = 0; i < n_nodes; i++){
      #pragma HLS unroll
      // Only non-leaf nodes do comparisons
      // negative values mean is a leaf (sklearn: -2)
      if(feature[i] >= 0){
        comparison[i] = x[feature[i]] <= threshold[i];
      }else{
        comparison[i] = true;
      }
    }

    // Determine node activity for all nodes
    int iLeaf = 0;
    Activate: for(int i = 0; i < n_nodes; i++){
      #pragma HLS unroll
      // Root node is always active
      if(i == 0){
        activation[i] = true;
      }else{
        // If this node is the left child of its parent
        if(i == children_left[parent[i]]){
          activation[i] = comparison[parent[i]] && activation[parent[i]];
        }else{ // Else it is the right child
          activation[i] = !comparison[parent[i]] && activation[parent[i]];
        }
      }
      // Skim off the leaves
      if(children_left[i] == -1){ // is a leaf
        activation_leaf[iLeaf] = activation[i];
        value_leaf[iLeaf] = value[i];
        iLeaf++;
      }
    }

    score_t y = 0;
    for(int i = 0; i < n_leaves; i++){
      if(activation_leaf[i]){
        return value_leaf[i];
      }
    }
    return y;
  }
};

template<int n_trees, int n_classes, class input_t, class score_t, class threshold_t>
struct BDT{

public:
  score_t normalisation;
  score_t init_predict[fn_classes(n_classes)];
  OpAdd<score_t> op_add;

  void tree_scores(input_t x, score_t scores[fn_classes(n_classes)][n_trees]) const;

  void decision_function(input_t x, score_t score[fn_classes(n_classes)]) const{
    score_t scores[fn_classes(n_classes)][n_trees];
    #pragma HLS ARRAY_PARTITION variable=scores dim=0
    // Get predictions scores
    tree_scores(x, scores);
    // Reduce
    Reduce:
    for(int j = 0; j < fn_classes(n_classes); j++){
      // Init predictions
      score[j] = init_predict[j];
      // Sum predictions from trees via "reduce" method
      score[j] += reduce<score_t, n_trees, OpAdd<score_t>>(scores[j], op_add);
    }
    // Normalize predictions
    for(int j = 0; j < fn_classes(n_classes); j++){
      score[j] *= normalisation;
    }
  }

};

}
#endif
