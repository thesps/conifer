// Copyright CERN 2026.
//
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2
// You may redistribute and modify this documentation and make products
// using it under the terms of the CERN-OHL-P v2 (https:/cern.ch/cern-ohl).
//
// This code is distributed WITHOUT ANY EXPRESS OR IMPLIED
// WARRANTY, INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY
// AND FITNESS FOR A PARTICULAR PURPOSE. Please see the CERN-OHL-P v2
// for applicable conditions
//
// Source location: https://github.com/thesps/conifer

// Verilog shim for the HLS RTL blackbox FPU_df_bb (see fpu_bb.h), instantiating the VHDL FPU_df_hls.
// HLS takes only Verilog blackbox RTL. The ports are those HLS connects for the FPU_df_bb arguments:
// x_i for each feature, y, then roots_i and nodes_i memory ports for each Tree Engine, concatenated for FPU_df_hls
// The ports and their connections are written by the conifer FPU writer
module FPU_df_bb(
  input ap_clk,
  input ap_rst,
  input ap_ce,
  input ap_start,
  output ap_done,
  output ap_idle,
  output ap_ready,
  input ap_continue,
  output [31:0] y,
  output y_ap_vld,
  // conifer insert ports
);

FPU_df_hls df (
  .ap_clk(ap_clk),
  .ap_rst(ap_rst),
  .ap_ce(ap_ce),
  .ap_start(ap_start),
  .ap_done(ap_done),
  .ap_idle(ap_idle),
  .ap_ready(ap_ready),
  .ap_continue(ap_continue),
  .y(y),
  .y_ap_vld(y_ap_vld),
  // conifer insert connections
);

endmodule
