# Out-of-context synthesis, and optionally implementation, of BDTTop
# BDTTop is wrapped in input and output registers (BDTTopOOC) so all of its paths are timed register to register
# part and clock_period are set by build_parameters.tcl, written by the conifer VHDL backend
# the build steps are selected by build_opt.tcl, written by the conifer VHDL backend build function
source build_parameters.tcl
source build_opt.tcl

add_files firmware/
remove_files firmware/SimulationInput.vhd
remove_files firmware/SimulationOutput.vhd
remove_files firmware/BDTTestbench.vhd
remove_files firmware/TestUtil.vhd
set_property file_type {VHDL 2008} [get_files *.vhd]
read_xdc -mode out_of_context build.xdc

synth_design -top BDTTopOOC -part $part -mode out_of_context

# place the clock source near the middle of the device for clock delay and skew estimation
set clk_sites [get_sites -quiet -filter {SITE_TYPE == BUFGCE}]
if {[llength $clk_sites] == 0} {
  set clk_sites [get_sites -quiet -filter {SITE_TYPE == BUFGCTRL}]
}
if {[llength $clk_sites] > 0} {
  set_property HD.CLK_SRC [lindex $clk_sites [expr {[llength $clk_sites] / 2}]] [get_ports clk]
}

report_timing_summary -max_paths 10 -file timing_synth.rpt
report_utilization -cells [get_cells BDTTop_inst] -file util.rpt
# writes netlist/BDTTop/BDTTop.edn
write_edif -cell BDTTop_inst -force netlist

if {$opt(impl)} {
  opt_design
  place_design
  phys_opt_design
  route_design

  report_timing_summary -max_paths 10 -file timing.rpt
  report_timing -max_paths 100 -nworst 1 -unique_pins -file timing_paths.rpt
  report_utilization -cells [get_cells BDTTop_inst] -file util_impl.rpt
  write_checkpoint -force post_route.dcp
}
