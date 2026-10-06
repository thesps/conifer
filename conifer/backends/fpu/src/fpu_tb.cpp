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
* Forest Processing Unit testbench
* Loads a packed model and executes inference on a batch of inputs, written by FPUBuilder.write_testbench_data
* Writes the predictions to tb_data/y.dat, and the model read back from the FPU to tb_data/roots_out.dat and tb_data/nodes_out.dat
*/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "fpu.h"
#include "parameters.h"

void FPU_TOP(int* X, int* y, int instruction, int batch_size, int n_features, int roots_in[NTE][NROOTS+1], int roots_out[NTE][NROOTS+1], InterfaceDecisionNode nodes_in[NTE][NNODES], InterfaceDecisionNode nodes_out[NTE][NNODES], float scales_in[NFEATURES+NCLASSES], float scales_out[NFEATURES+NCLASSES], char* info, int& infoLength);

int main(){
  static int roots[NTE][NROOTS+1];
  static int roots_out[NTE][NROOTS+1];
  static InterfaceDecisionNode nodes[NTE][NNODES];
  static InterfaceDecisionNode nodes_out[NTE][NNODES];
  float scales[NFEATURES+NCLASSES];
  float scales_out[NFEATURES+NCLASSES];
  char info[theInfoLength];
  int infoLength;
  int dummy[1] = {0};

  std::ifstream f_roots("tb_data/roots.dat");
  std::ifstream f_nodes("tb_data/nodes.dat");
  std::ifstream f_scales("tb_data/scales.dat");
  std::ifstream f_X("tb_data/X.dat");
  if(!f_roots.is_open() || !f_nodes.is_open() || !f_scales.is_open() || !f_X.is_open()){
    std::cerr << "ERROR: could not open testbench data in tb_data" << std::endl;
    return 1;
  }
  for(int i = 0; i < NTE; i++){
    for(int j = 0; j < NROOTS+1; j++){
      f_roots >> roots[i][j];
    }
    for(int j = 0; j < NNODES; j++){
      InterfaceDecisionNode& n = nodes[i][j];
      f_nodes >> n.threshold >> n.score >> n.feature >> n.child_left >> n.child_right >> n.iclass >> n.is_leaf;
    }
  }
  for(int i = 0; i < NFEATURES+NCLASSES; i++){
    f_scales >> scales[i];
  }
  int batch_size, n_features;
  f_X >> batch_size >> n_features;
  std::vector<int> X(batch_size * n_features);
  std::vector<int> y(batch_size);
  for(int i = 0; i < batch_size * n_features; i++){
    if(SCALER){
      float x;
      f_X >> x;
      X[i] = *reinterpret_cast<int*>(&x);
    }else{
      f_X >> X[i];
    }
  }

  // load the model
  FPU_TOP(dummy, dummy, 1, 0, 0, roots, roots_out, nodes, nodes_out, scales, scales_out, info, infoLength);
  // read the model back
  FPU_TOP(dummy, dummy, 2, 0, 0, roots, roots_out, nodes, nodes_out, scales, scales_out, info, infoLength);
  // inference
  FPU_TOP(X.data(), y.data(), 3, batch_size, n_features, roots, roots_out, nodes, nodes_out, scales, scales_out, info, infoLength);

  std::ofstream f_y("tb_data/y.dat");
  f_y << std::setprecision(9);
  for(int i = 0; i < batch_size; i++){
    if(SCALER){
      f_y << *reinterpret_cast<float*>(&y[i]) << "\n";
    }else{
      f_y << y[i] << "\n";
    }
  }
  f_y.close();

  std::ofstream f_roots_out("tb_data/roots_out.dat");
  std::ofstream f_nodes_out("tb_data/nodes_out.dat");
  for(int i = 0; i < NTE; i++){
    for(int j = 0; j < NROOTS+1; j++){
      f_roots_out << roots_out[i][j] << (j < NROOTS ? " " : "\n");
    }
    for(int j = 0; j < NNODES; j++){
      InterfaceDecisionNode& n = nodes_out[i][j];
      f_nodes_out << n.threshold << " " << n.score << " " << n.feature << " " << n.child_left << " " << n.child_right << " " << n.iclass << " " << n.is_leaf << "\n";
    }
  }
  f_roots_out.close();
  f_nodes_out.close();
  std::cout << "Wrote " << batch_size << " predictions to tb_data/y.dat" << std::endl;
  return 0;
}
