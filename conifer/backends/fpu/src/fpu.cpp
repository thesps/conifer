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

/*
* Forest Processing Unit top level
*/

#include "fpu.h"
#include "parameters.h"
#include <cstdint>

void FPU_internal(int* X, int* y, int instruction, int batch_size, int n_features, InterfaceDecisionNode nodes_in[NTE][NNODES], InterfaceDecisionNode nodes_out[NTE][NNODES], float scales_in[NFEATURES+NCLASSES], float scales_out[NFEATURES+NCLASSES], char* info, int& infoLength){
  static DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes_int[NTE][NNODES];
  #pragma HLS array_partition variable=nodes_int dim=1
  #pragma HLS aggregate variable=nodes_int compact=bit  
  static float scales_int[NFEATURES+NCLASSES];

  infoLength = theInfoLength;

  if(instruction == 0){
    for(int i = 0; i < theInfoLength; i++){
      info[i] = theInfo[i];
    }
  }
  if(instruction == 1){
    LoadTE: for(int i = 0; i < NTE; i++){
      LoadNode: for(int j = 0; j < NNODES; j++){
        nodes_int[i][j].fromInterfaceNode(nodes_in[i][j]);
      }
    }
    LoadScales: for(int i = 0; i < NFEATURES + NCLASSES; i++){
      scales_int[i] = scales_in[i];
    }
  }
  if(instruction == 2){
    ReadTE: for(int i = 0; i < NTE; i++){
      ReadNode: for(int j = 0; j < NNODES; j++){
        nodes_out[i][j] = nodes_int[i][j].toInterfaceNode();
      }
    }
    ReadScales: for(int i = 0; i < NFEATURES + NCLASSES; i++){
      scales_out[i] = scales_int[i];
    }    
  }
  if(instruction == 3){
    T X_int[NFEATURES];
    U y_int;
    #pragma HLS array_partition variable=X_int
    for(int n = 0; n < batch_size; n++){
      for(int i = 0; i < n_features; i++){
        if(SCALER){
          float x_float = *reinterpret_cast<float*>(&X[n*n_features + i]);
          float x_float_scaled = x_float * scales_int[i];
          X_int[i] = (T) x_float_scaled;
        }else{
          X_int[i] = (T) X[n*n_features + i];
        }
      }
      FPU_df<T, U, FEATBITS, ADDRBITS, CLASSBITS, NFEATURES, NNODES, NTE>(X_int, y_int, nodes_int);
      if(SCALER){
        float y_tmp = ((float) y_int) * scales_int[NFEATURES];
        y[n] = *(reinterpret_cast<int*>(&y_tmp));
      }else{
        y[n] = y_int;
      }
    }
  }
}

void FPU_Zynq(int* X, int* y, int instruction, int batch_size, int n_features, InterfaceDecisionNode nodes_in[NTE][NNODES], InterfaceDecisionNode nodes_out[NTE][NNODES], float scales_in[NFEATURES+NCLASSES], float scales_out[NFEATURES+NCLASSES], char* info, int& infoLength){
  #pragma HLS INTERFACE mode=m_axi port=X offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=y offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=info offset=slave bundle=gmem0

  #pragma HLS INTERFACE mode=s_axilite port=instruction bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=batch_size bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=n_features bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=X bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=y bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=nodes_in bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=nodes_out bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=scales_in bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=scales_out bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=info bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=infoLength bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  FPU_internal(X, y, instruction, batch_size, n_features, nodes_in, nodes_out, scales_in, scales_out, info, infoLength);
}

void FPU_Alveo(int* X, int* y, int instruction, int batch_size, int n_features, InterfaceDecisionNode nodes_in[NTE][NNODES], InterfaceDecisionNode nodes_out[NTE][NNODES], float scales_in[NFEATURES+NCLASSES], float scales_out[NFEATURES+NCLASSES], char* info, int& infoLength){
  #pragma HLS INTERFACE mode=m_axi port=X offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=y offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=info offset=slave bundle=gmem0
  FPU_internal(X, y, instruction, batch_size, n_features, nodes_in, nodes_out, scales_in, scales_out, info, infoLength);
}