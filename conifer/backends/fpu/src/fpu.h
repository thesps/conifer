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

/*
* Packed node word, for the VHDL Tree Engines (see vhdl/fpu_pkg.vhd). From the least significant bit:
* threshold | score | feature | child_left | child_right | iclass | is_leaf
*/
template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS>
struct NodeWord{
  static const int TBITS = T::width;
  static const int UBITS = U::width;
  static const int BITS = TBITS + UBITS + FEATBITS + 2 * ADDRBITS + CLASSBITS + 1;
  typedef ap_uint<BITS> word;
  typedef DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> DN;

  static word pack(const DN& n){
    word w = 0;
    int lo = 0;
    w.range(lo + TBITS - 1, lo) = n.threshold.range(TBITS - 1, 0); lo += TBITS;
    w.range(lo + UBITS - 1, lo) = n.score.range(UBITS - 1, 0); lo += UBITS;
    w.range(lo + FEATBITS - 1, lo) = n.feature.range(FEATBITS - 1, 0); lo += FEATBITS;
    w.range(lo + ADDRBITS - 1, lo) = n.child_left.range(ADDRBITS - 1, 0); lo += ADDRBITS;
    w.range(lo + ADDRBITS - 1, lo) = n.child_right.range(ADDRBITS - 1, 0); lo += ADDRBITS;
    w.range(lo + CLASSBITS - 1, lo) = n.iclass.range(CLASSBITS - 1, 0); lo += CLASSBITS;
    w[lo] = n.is_leaf;
    return w;
  }

  static DN unpack(const word& w){
    DN n;
    int lo = 0;
    n.threshold.range(TBITS - 1, 0) = w.range(lo + TBITS - 1, lo); lo += TBITS;
    n.score.range(UBITS - 1, 0) = w.range(lo + UBITS - 1, lo); lo += UBITS;
    n.feature.range(FEATBITS - 1, 0) = w.range(lo + FEATBITS - 1, lo); lo += FEATBITS;
    n.child_left.range(ADDRBITS - 1, 0) = w.range(lo + ADDRBITS - 1, lo); lo += ADDRBITS;
    n.child_right.range(ADDRBITS - 1, 0) = w.range(lo + ADDRBITS - 1, lo); lo += ADDRBITS;
    n.iclass.range(CLASSBITS - 1, 0) = w.range(lo + CLASSBITS - 1, lo); lo += CLASSBITS;
    n.is_leaf = w[lo];
    return n;
  }
};

template<class T, class U, int FEATBITS, int ADDRBITS, int CLASSBITS, int NVARS, int NNODES, int NROOTS>
U TreeEngine(T X[NVARS], ap_uint<ADDRBITS> roots[NROOTS+1], typename NodeWord<T,U,FEATBITS,ADDRBITS,CLASSBITS>::word nodes[NNODES]){
  // C model of the VHDL Tree Engine (vhdl/TreeEngine.vhd), on its packed memories: walk the trees one after another.
  // The VHDL walks two trees at a time, which gives the same sum
  typedef NodeWord<T,U,FEATBITS,ADDRBITS,CLASSBITS> NW;
  U y = 0;
  int n_roots = roots[0];
  for(int r = 0; r < n_roots && r < NROOTS; r++){
    int i = roots[r + 1];
    auto node = NW::unpack(nodes[i]);
    while(!node.is_leaf){
      i = X[node.feature] <= node.threshold ? node.child_left : node.child_right;
      node = NW::unpack(nodes[i]);
    }
    y += node.score;
  }
  return y;
}

template<class T>
T dynamic_scaler(float x, float s){
  #pragma HLS pipeline
  float y_f = x * s;
  return (T) y_f;
} 

#endif