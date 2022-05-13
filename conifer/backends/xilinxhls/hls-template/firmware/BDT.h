#ifndef BDT_H__
#define BDT_H__

#include "ap_fixed.h"

namespace BDT{

constexpr int pow2(int x){
  return x == 0 ? 1 : 2 * pow2(x - 1);
}

   //hls-fpga-machine-learning insert switch case nodes
}

   //hls-fpga-machine-learning insert switch case leaves
}

constexpr int fn_classes(int n_classes){
  // Number of trees given number of classes
  return n_classes == 2 ? 1 : n_classes;
}

template<int tree_idx, class input_t, class score_t, class threshold_t>
struct Tree {
private:
  static constexpr int n_nodes = fn_nodes(tree_idx);
  static constexpr int n_leaves = fn_leaves(tree_idx);
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

template<int n_trees, int max_depth, int n_classes, class input_t, class score_t, class threshold_t, bool unroll>
struct BDT{

public:
    score_t normalisation;
	score_t init_predict[fn_classes(n_classes)];
        //hls-fpga-machine-learning insert trees
	void decision_function(input_t x, score_t score[fn_classes(n_classes)], score_t tree_scores[fn_classes(n_classes) * n_trees]) const{
        if(unroll){
//    		#pragma HLS ARRAY_PARTITION variable=trees dim=0
        }
		for(int j = 0; j < fn_classes(n_classes); j++){
			score[j] = init_predict[j];
		}
		Trees:
        //hls-fpga-machine-learning insert decision functions
        for(int j = 0; j < fn_classes(n_classes); j++){
            score[j] *= normalisation;
        }
	}

};

//hls-fpga-machine-learning insert declarations

}
#endif
