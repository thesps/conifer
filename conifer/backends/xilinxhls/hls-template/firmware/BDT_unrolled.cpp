#include "BDT.h"
#include "parameters.h"

template<>
void BDT::BDT<n_trees, n_classes, input_arr_t, score_t, threshold_t>::tree_scores(input_arr_t x, score_t scores[fn_classes(n_classes)][n_trees]) const {
  // conifer insert tree_scores
}

