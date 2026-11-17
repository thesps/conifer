#!/bin/sh
# Simulate an FPU testbench with xsim, in the vhdl directory of an FPUBuilder project, on the testbench data in ../tb_data
# written by FPUBuilder.write_testbench_data for an FPU without the dynamic scaler.
# Usage: xsim_tb.sh [testbench [VHDL files]]: the testbench entity, default FPU_df_tb, in testbench.vhd, and the VHDL
# files it needs other than the core's, in compile order. FPU_df_tb writes the predictions to y.dat
set -e
cd "$(dirname "$0")"
TB=${1:-FPU_df_tb}
[ $# -gt 0 ] && shift
xvhdl -2008 fpu_pkg.vhd TreeEngine.vhd FPU_df.vhd "$@" $TB.vhd
xelab -debug off -s ${TB}_sim $TB
xsim ${TB}_sim -R
