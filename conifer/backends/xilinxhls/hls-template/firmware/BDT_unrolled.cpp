#include "BDT.h"
#include "parameters.h"

bool (*split_fn)(const input_t*, const threshold_t*) = !strcmp(splitting_convention,"<=") ? [](const input_t *a, const threshold_t *b) { return *a <= *b; } : [](const input_t *a, const threshold_t *b) { return *a < *b;};

template<>
void BDT::BDT<n_trees, n_classes, input_arr_t, score_t, threshold_t>::tree_scores(input_arr_t x, score_t scores[fn_classes(n_classes)][n_trees]) const {
  // conifer insert tree_scores
}

