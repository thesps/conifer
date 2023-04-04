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

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS, int NVARS, int NNODES>
void TreeEngine(T X[NVARS], DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes[NNODES], U& y){
  #pragma HLS pipeline
  ap_int<ADDRBITS> i = 0;
  auto node = nodes[i];
  node_loop : while(!node.is_leaf){
    #pragma HLS pipeline
    i = X[node.feature] <= node.threshold ? node.child_left : node.child_right;
    node = nodes[i];
  }
  y = node.score;
}

template<class T>
T dynamic_scaler(float x, float s){
  #pragma HLS pipeline
  float y_f = x * s;
  return (T) y_f;
} 

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS, int NVARS, int NNODES, int NTE>
void FPU_df(T X[NVARS], U& y, DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes[NTE][NNODES]){
    U y_acc = 0;
    for(int i = 0; i < NTE; i++){
      #pragma HLS unroll
      U y_i = 0;
      TreeEngine<T, U, FEATBITS, ADDRBITS, CLASSBITS, NVARS, NNODES>(X, nodes[i], y_i);
      y_acc += y_i;
    }
    y = y_acc;
}

#endif