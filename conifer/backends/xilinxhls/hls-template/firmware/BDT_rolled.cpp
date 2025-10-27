#include "BDT.h"
#include "parameters.h"

bool (*split_fn)(const input_t*, const threshold_t*) = !strcmp(splitting_convention,"<=") ? [](const input_t *a, const threshold_t *b) { return *a <= *b; } : [](const input_t *a, const threshold_t *b) { return *a < *b;};

template<>
void BDT::BDT<n_trees, max_depth, n_classes, n_features, input_arr_t, score_t, weight_t, threshold_t, unroll>::tree_scores(input_arr_t x, score_t scores[n_trees][fn_classes(n_classes)]) const {
  Trees:
  for(int i = 0; i < n_trees; i++){
    Classes:
    for(int j = 0; j < fn_classes(n_classes); j++){
      scores[i][j] = trees[i][j].decision_function(x, split_fn);
    }
  }
}
