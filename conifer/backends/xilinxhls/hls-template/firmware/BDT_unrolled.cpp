#include "BDT.h"
#include "parameters.h"

std::function<bool (input_t, threshold_t)> split_fn = !strcmp(splitting_convention,"<=") ? [](const input_t &a, const threshold_t &b) { return a <= b; } : [](const input_t &a, const threshold_t &b) { return a < b;};

template<>
void BDT::BDT<n_trees, n_classes, input_arr_t, score_t, threshold_t>::tree_scores(input_arr_t x, score_t scores[fn_classes(n_classes)][n_trees]) const {
  // conifer insert tree_scores
}

