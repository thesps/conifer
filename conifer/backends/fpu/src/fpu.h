/*
Copyright CERN 2023.

This source describes Open Hardware and is licensed under the CERN-OHL-P v2
You may redistribute and modify this documentation and make products
using it under the terms of the CERN-OHL-P v2 (https:/cern.ch/cern-ohl).

This code is distributed WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY, INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY
AND FITNESS FOR A PARTICULAR PURPOSE. Please see the CERN-OHL-P v2
for applicable conditions

Source location: https://github.com/thesps/conifer
*/

#ifndef CONIFER_FPU_H__
#define CONIFER_FPU_H__

#include "ap_fixed.h"

struct InterfaceDecisionNode{
  int threshold;
  int score;
  int feature;
  int child_left;
  int child_right;
  int iclass;
  int is_leaf;
};

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS>
struct DecisionNode{
  T threshold;
  U score;
  ap_int<FEATBITS> feature;
  ap_int<ADDRBITS> child_left;
  ap_int<ADDRBITS> child_right;
  ap_int<CLASSBITS> iclass;
  bool is_leaf;

  void fromInterfaceNode(InterfaceDecisionNode n){
    this->threshold = n.threshold;
    this->score = n.score;
    this->feature = n.feature;
    this->child_left = n.child_left;
    this->child_right = n.child_right;
    this->iclass = n.iclass;
    this->is_leaf = n.is_leaf;
  }

  InterfaceDecisionNode toInterfaceNode(){
    InterfaceDecisionNode n;
    n.threshold = this->threshold;
    n.score = this->score;
    n.feature = this->feature;
    n.child_left = this->child_left;
    n.child_right = this->child_right;
    n.iclass = this->iclass;
    n.is_leaf = this->is_leaf;
    return n;
  }
};

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS, int NVARS, int NNODES, int NROOTS>
void TreeEngine(T X[NVARS], ap_int<ADDRBITS> roots[NROOTS+1], DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes[NNODES], U& y){
  // roots[0] is the number of trees loaded into this TE, roots[1:roots[0]+1] are the address of each tree's root node
  // Walk the trees one after another in a single loop, moving on to the next root on reaching a leaf
  int n_roots = roots[0];
  int r = 1;
  ap_int<ADDRBITS> i_next = roots[NROOTS > 1 ? 2 : 1];
  bool last = n_roots <= 1;
  bool done = n_roots == 0;
  U y_acc = 0;
  auto node = nodes[roots[1]];
  node_loop : while(!done){
    #pragma HLS pipeline
    #pragma HLS loop_tripcount min=0 max=NNODES
    // select the next root in parallel with the comparison, leaving one mux after the comparison as for a single tree
    ap_int<ADDRBITS> next_left = node.is_leaf ? i_next : node.child_left;
    ap_int<ADDRBITS> next_right = node.is_leaf ? i_next : node.child_right;
    ap_int<ADDRBITS> i = X[node.feature] <= node.threshold ? next_left : next_right;
    if(node.is_leaf){
      y_acc += node.score;
      done = last;
      r++;
      last = r >= n_roots;
      i_next = roots[r < NROOTS ? r + 1 : NROOTS];
    }
    node = nodes[i];
  }
  y = y_acc;
}

template<class T>
T dynamic_scaler(float x, float s){
  #pragma HLS pipeline
  float y_f = x * s;
  return (T) y_f;
} 

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS, int NVARS, int NNODES, int NTE, int NROOTS>
void FPU_df(T X[NVARS], U& y, ap_int<ADDRBITS> roots[NTE][NROOTS+1], DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes[NTE][NNODES]){
    U y_acc = 0;
    for(int i = 0; i < NTE; i++){
      #pragma HLS unroll
      U y_i = 0;
      TreeEngine<T, U, FEATBITS, ADDRBITS, CLASSBITS, NVARS, NNODES, NROOTS>(X, roots[i], nodes[i], y_i);
      y_acc += y_i;
    }
    y = y_acc;
}

#endif