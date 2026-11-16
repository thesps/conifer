#!/bin/sh
# Simulate FPU_df_tb with xsim, in the vhdl directory of an FPUBuilder project, on the testbench data in ../tb_data
# written by FPUBuilder.write_testbench_data for an FPU without the dynamic scaler. Writes the predictions to y.dat
set -e
cd "$(dirname "$0")"
xvhdl -2008 fpu_pkg.vhd TreeEngine.vhd FPU_df.vhd FPU_df_tb.vhd
xelab -debug off -s FPU_df_tb_sim FPU_df_tb
xsim FPU_df_tb_sim -R
