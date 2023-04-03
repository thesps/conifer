#include "BDT.h"
#include "parameters.h"

template<>
void BDT::BDT<n_trees, max_depth, n_classes, input_arr_t, score_t, threshold_t, unroll>::tree_scores(input_arr_t x, score_t scores[n_trees][fn_classes(n_classes)]) const {
  Trees:
  for(int i = 0; i < n_trees; i++){
    Classes:
    for(int j = 0; j < fn_classes(n_classes); j++){
      scores[i][j] = trees[i][j].decision_function(x);
    }
  }
}