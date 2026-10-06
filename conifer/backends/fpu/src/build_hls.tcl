# Copyright CERN 2023.
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

set tcldir [file dirname [info script]]
source [file join $tcldir hls_parameters.tcl]

open_project -reset ${prj_name}

set_top ${top}

if {${cosim}} {
  add_files fpu.cpp -cflags "-std=c++17 -DCOSIM_X_DEPTH=${cosim_X_depth} -DCOSIM_Y_DEPTH=${cosim_y_depth} -DCOSIM_INFO_DEPTH=${cosim_info_depth}"
} else {
  add_files fpu.cpp -cflags "-std=c++17"
}
if {${csim} || ${cosim}} {
  add_files -tb fpu_tb.cpp -cflags "-std=c++17 -DFPU_TOP=${top}"
  add_files -tb tb_data
}

open_solution -reset "solution1" -flow_target ${flow_target}

set_part ${part}
create_clock -period ${clock_period} -name clk

config_interface -m_axi_addr64=${m_axi_addr64}

if {${csim}} {
  csim_design
}
csynth_design
if {${cosim}} {
  cosim_design
}
export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
quit
