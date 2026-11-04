/*
Copyright CERN 2026.

This source describes Open Hardware and is licensed under the CERN-OHL-P v2
You may redistribute and modify this documentation and make products
using it under the terms of the CERN-OHL-P v2 (https:/cern.ch/cern-ohl).

This code is distributed WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY, INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY
AND FITNESS FOR A PARTICULAR PURPOSE. Please see the CERN-OHL-P v2
for applicable conditions

Source location: https://github.com/thesps/conifer
*/

// C model of the VHDL FPU_df RTL blackbox FPU_df_bb, used for C simulation
#define FPU_BB_MODEL
#include "fpu_bb.h"

#define FPU_BB_X_SET(i) X[i].range(NW::TBITS - 1, 0) = x_##i;
#define FPU_BB_ROOTS_PTR(i) roots_##i,
#define FPU_BB_NODES_PTR(i) nodes_##i,

void FPU_df_bb(FPU_BB_PARAMS){
  T X[NFEATURES];
  FPU_FOR_EACH_FEATURE(FPU_BB_X_SET)
  RootWord* roots[NTE] = {FPU_FOR_EACH_TE(FPU_BB_ROOTS_PTR)};
  NW::word* nodes[NTE] = {FPU_FOR_EACH_TE(FPU_BB_NODES_PTR)};
  U acc = 0;
  for(int i = 0; i < NTE; i++){
    acc += TreeEngine<T, U, FEATBITS, ADDRBITS, CLASSBITS, NFEATURES, NNODES, NROOTS>(X, roots[i], nodes[i]);
  }
  // the score, sign extended to the 32 bit interface
  ap_int<NW::UBITS> y_bits;
  y_bits.range() = acc.range(NW::UBITS - 1, 0);
  y = y_bits.to_int();
}
