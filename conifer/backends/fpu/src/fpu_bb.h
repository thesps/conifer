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

/*
* The VHDL FPU_df RTL blackbox FPU_df_bb, and the Tree Engine memories it reads.
* HLS passes only one dimensional arrays, and int scalars, to a blackbox. So each feature of X is an int argument,
* and each Tree Engine has its own node and roots memory arrays. These are listed with the macros
* FPU_FOR_EACH_FEATURE(M) and FPU_FOR_EACH_TE(M) from parameters.h, which expand M(i) for each feature (Tree Engine) i.
*/
#ifndef CONIFER_FPU_BB_H_
#define CONIFER_FPU_BB_H_
#include "fpu.h"
#include "parameters.h"

typedef NodeWord<T,U,FEATBITS,ADDRBITS,CLASSBITS> NW;
typedef ap_uint<ADDRBITS> RootWord;

// FPU_df_bb(int x_0, ..., int& y, RootWord roots_0[], ..., NW::word nodes_0[], ...)
#define FPU_BB_X_PARAM(i) int x_##i,
#define FPU_BB_ROOTS_PARAM(i) , RootWord roots_##i[NROOTS+1]
#define FPU_BB_NODES_PARAM(i) , NW::word nodes_##i[NNODES]
#define FPU_BB_PARAMS FPU_FOR_EACH_FEATURE(FPU_BB_X_PARAM) int& y FPU_FOR_EACH_TE(FPU_BB_ROOTS_PARAM) FPU_FOR_EACH_TE(FPU_BB_NODES_PARAM)

void FPU_df_bb(FPU_BB_PARAMS);

#ifndef FPU_BB_MODEL

// The Tree Engine memories: fpu_nodes_i, fpu_roots_i
#define FPU_TE_MEMORIES(i) static NW::word fpu_nodes_##i[NNODES]; static RootWord fpu_roots_##i[NROOTS+1];
FPU_FOR_EACH_TE(FPU_TE_MEMORIES)

// One loop over all Tree Engines, writing (reading) the Tree Engine memory selected by a switch.
// The interface is read before (written after) the switch, so there is one interface access, not one per case
#define FPU_LOAD_NODES_CASE(i) case i: fpu_nodes_##i[j] = w; break;
#define FPU_LOAD_ROOTS_CASE(i) case i: fpu_roots_##i[j] = r; break;
#define FPU_READ_NODES_CASE(i) case i: w = fpu_nodes_##i[j]; break;
#define FPU_READ_ROOTS_CASE(i) case i: r = fpu_roots_##i[j]; break;

static void FPU_vhdl_load(InterfaceDecisionNode nodes_in[NTE][NNODES], int roots_in[NTE][NROOTS+1]){
  LoadTE: for(int i = 0; i < NTE; i++){
    LoadNode: for(int j = 0; j < NNODES; j++){
      DN n;
      n.fromInterfaceNode(nodes_in[i][j]);
      NW::word w = NW::pack(n);
      switch(i){
        FPU_FOR_EACH_TE(FPU_LOAD_NODES_CASE)
      }
    }
  }
  LoadRootsTE: for(int i = 0; i < NTE; i++){
    LoadRoots: for(int j = 0; j < NROOTS+1; j++){
      int r = roots_in[i][j];
      switch(i){
        FPU_FOR_EACH_TE(FPU_LOAD_ROOTS_CASE)
      }
    }
  }
}

static void FPU_vhdl_read(InterfaceDecisionNode nodes_out[NTE][NNODES], int roots_out[NTE][NROOTS+1]){
  ReadRootsTE: for(int i = 0; i < NTE; i++){
    ReadRoots: for(int j = 0; j < NROOTS+1; j++){
      int r = 0;
      switch(i){
        FPU_FOR_EACH_TE(FPU_READ_ROOTS_CASE)
      }
      roots_out[i][j] = r;
    }
  }
  ReadTE: for(int i = 0; i < NTE; i++){
    ReadNode: for(int j = 0; j < NNODES; j++){
      NW::word w = 0;
      switch(i){
        FPU_FOR_EACH_TE(FPU_READ_NODES_CASE)
      }
      nodes_out[i][j] = NW::unpack(w).toInterfaceNode();
    }
  }
}

// Call the blackbox. It must be called from a function that is not inlined, so it is not in a pipeline
#define FPU_BB_X_ARG(i) X[i].range(NW::TBITS - 1, 0).to_int(),
#define FPU_BB_ROOTS_ARG(i) , fpu_roots_##i
#define FPU_BB_NODES_ARG(i) , fpu_nodes_##i

static void FPU_df_vhdl(T X[NFEATURES], U& y){
  #pragma HLS inline off
  int y_bb;
  FPU_df_bb(FPU_FOR_EACH_FEATURE(FPU_BB_X_ARG) y_bb FPU_FOR_EACH_TE(FPU_BB_ROOTS_ARG) FPU_FOR_EACH_TE(FPU_BB_NODES_ARG));
  y.range(NW::UBITS - 1, 0) = ap_int<32>(y_bb).range(NW::UBITS - 1, 0);
}

#endif
#endif
