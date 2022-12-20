/*
* Forest Processing Unit top level
*/

#include "fpu.h"
#include "parameters.h"

void FPU(int* X, int* y, int instruction, InterfaceDecisionNode nodes_in[NTE][NNODES], InterfaceDecisionNode nodes_out[NTE][NNODES], float scales_in[NFEATURES], float scales_out[NFEATURES], char* info, int& infoLength){
  #pragma HLS INTERFACE mode=m_axi port=X offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=y offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=nodes_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_in offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axi port=scales_out offset=slave bundle=gmem0
  #pragma HLS INTERFACE mode=m_axis port=info offset=slave bundle=gmem0

  #pragma HLS INTERFACE mode=s_axilite port=instruction bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=X bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=y bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=nodes_in bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=nodes_out bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=scales_in bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=scales_out bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=info bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=infoLength bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

  static DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> nodes_int[NTE][NNODES];
  #pragma HLS array_partition variable=nodes_int dim=1
  #pragma HLS aggregate variable=nodes_int compact=bit  
  static float scales_int[NFEATURES];

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
    LoadScales: for(int i = 0; i < NFEATURES; i++){
      scales_int[i] = scales_in[i];
    }
  }
  if(instruction == 2){
    ReadTE: for(int i = 0; i < NTE; i++){
      ReadNode: for(int j = 0; j < NNODES; j++){
        nodes_out[i][j] = nodes_int[i][j].toInterfaceNode();
      }
    }
    ReadScales: for(int i = 0; i < NFEATURES; i++){
      scales_out[i] = scales_int[i];
    }    
  }
  if(instruction == 3){
    T X_int[NFEATURES];
    U y_int;
    #pragma HLS array_partition variable=X_int
    /*ApplyScales: for(int i = 0; i < NFEATURES; i++){
      #pragma HLS pipeline
      if(SCALER){
        X_int[i] = dynamic_scaler<T>(X[i], scales_int[i]);
      }else{
        X_int[i] = X[i];
      }
    }*/
    for(int i = 0; i < NFEATURES; i++){
      X_int[i] = X[i];
    }
    FPU_df<T, U, FEATBITS, ADDRBITS, CLASSBITS, NFEATURES, NNODES, NTE>(X_int, y_int, nodes_int);
    y[0] = y_int;
  }
}