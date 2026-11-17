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

# Out of context synthesis of an FPU's VHDL: FPU_df_hls for the hls interface, FPU_rtl for the rtl interface
# Run in an FPUBuilder project: vivado -mode batch -source vhdl/synth_ooc.tcl
# The top, part, clocks, VHDL files and constraints are those of the project, from synth_parameters.tcl
# Writes utilization.rpt, timing.rpt and cdc.rpt to the vhdl directory, and prints the worst negative slack
set dir [file dirname [file normalize [info script]]]
source [file join $dir synth_parameters.tcl]
cd $dir
read_vhdl -vhdl2008 $vhdl_files
# the clocks are read before synthesis, so synthesis is timing driven
set f [open ooc_clock.xdc w]
foreach {name period} $clocks {
  puts $f "create_clock -name $name -period $period \[get_ports $name\]"
}
close $f
read_xdc -mode out_of_context ooc_clock.xdc
foreach x $xdc_files {
  read_xdc $x
}
synth_design -top $top -part $part -mode out_of_context
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
report_cdc -details -file cdc.rpt
puts "$top WNS [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] ns at period [lindex $clocks 1] ns"
