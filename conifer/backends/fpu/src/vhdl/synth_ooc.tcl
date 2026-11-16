# Copyright CERN 2026.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-P v2
# You may redistribute and modify this documentation and make products
# using it under the terms of the CERN-OHL-P v2 (https:/cern.ch/cern-ohl).
#
# This code is distributed WITHOUT ANY EXPRESS OR IMPLIED
# WARRANTY, INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY
# AND FITNESS FOR A PARTICULAR PURPOSE. Please see the CERN-OHL-P v2
# for applicable conditions
#
# Source location: https://github.com/thesps/conifer

# Out of context synthesis of the VHDL decision function as integrated in HLS (FPU_df_hls), without the HLS
# Run in an FPUBuilder project: vivado -mode batch -source vhdl/synth_ooc.tcl
# The part, clock period and VHDL files are those of the project, from hls_parameters.tcl
# Writes utilization.rpt and timing.rpt to the vhdl directory, and prints the worst negative slack
set dir [file dirname [file normalize [info script]]]
source [file join $dir .. hls_parameters.tcl]
cd $dir
read_vhdl -vhdl2008 $vhdl_files
# the clock is read before synthesis, so synthesis is timing driven
set f [open ooc_clock.xdc w]
puts $f "create_clock -name ap_clk -period $clock_period \[get_ports ap_clk\]"
close $f
read_xdc -mode out_of_context ooc_clock.xdc
synth_design -top FPU_df_hls -part $part -mode out_of_context
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
puts "FPU_df_hls WNS [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] ns at period $clock_period ns"
