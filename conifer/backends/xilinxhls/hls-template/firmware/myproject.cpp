#include "BDT.h"
#include "parameters.h"
#include "myproject.h"
#include "hls_stream.h"

void myproject(input_arr_t x, score_arr_t score){
  // conifer insert pragmas
  bdt.decision_function(x, score);
}

void copy_input(int n, accelerator_input_t* x_in, input_arr_t x_int){
  for(int i = 0; i < n_features; i++){
    x_int[i] = x_in[n_features*n + i];
  }
}

void copy_output(int n, score_arr_t score_int, accelerator_output_t* score_out){
  for(int i = 0; i < BDT::fn_classes(n_classes); i++){
    score_out[BDT::fn_classes(n_classes)*n + i] = score_int[i];
  }
}

void load(int N, accelerator_input_t* x, hls::stream<input_t>& x_stream){
  for(int n = 0; n < N; n++){
    for(int i = 0; i < n_features; i++){
      #pragma HLS pipeline
      input_t xi = x[n * n_features + i];
      x_stream.write(xi);
    }
  }
}

void compute(int N, hls::stream<input_t>& x_stream, hls::stream<score_t>& score_stream){
  for(int n = 0; n < N; n++){
    input_arr_t x_int;
    score_arr_t score_int;
    for(int i = 0; i < n_features; i++){
      #pragma HLS pipeline
      x_int[i] = x_stream.read();
    }
    myproject(x_int, score_int);
    for(int i = 0; i < BDT::fn_classes(n_classes); i++){
      #pragma HLS pipeline
      score_stream.write(score_int[i]);
    }
  }
}

void store(int N, hls::stream<score_t>& score_stream, accelerator_output_t* score){
  for(int n = 0; n < N; n++){
    for(int i = 0; i < BDT::fn_classes(n_classes); i++){
      #pragma HLS pipeline
      input_t scorei = score_stream.read();
      score[n * BDT::fn_classes(n_classes) + i] = scorei;
    }
  }
}

void myproject_accelerator(int N, int& n_f, int& n_c, accelerator_input_t* x, accelerator_output_t* score){
  // conifer insert accelerator pragmas
  #pragma HLS dataflow
  n_f = n_features;
  n_c = BDT::fn_classes(n_classes);

  hls::stream<input_t> x_stream("x_stream");
  hls::stream<score_t> score_stream("score_stream");
  #pragma HLS STREAM variable=x_stream depth=1024
  #pragma HLS STREAM variable=score_stream depth=1024

  load(N, x, x_stream);
  compute(N, x_stream, score_stream);
  store(N, score_stream, score);
}