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
# the decision function is the VHDL FPU_df, through the Verilog shim FPU_df_bb.v
add_files -blackbox FPU_df_bb.json
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
set sol_dir [file normalize ${prj_name}/solution1]
# A blackbox's RTL files may only be Verilog, so the VHDL instantiated by the shim is added to the generated RTL here
# Cosimulation compiles every file in syn/verilog
foreach f ${vhdl_files} {
  file copy -force vhdl/${f} ${sol_dir}/syn/verilog/${f}
}
if {${cosim}} {
  cosim_design
}
export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
# Repackage the IP with the VHDL added. export_design packages only the RTL listed under "Verilog" in the solution
# data <sol>_data.json, which does not include the VHDL of the blackbox. So: copy the VHDL to impl/verilog, add it
# to that list, and rerun the packaging. export_design generates the packaging in impl/ip: the script pack.sh runs
# Vivado on run_ippack.tcl, which reads <sol>_data.json and writes the IP (component.xml and the zip) in impl/ip
set sol_data_file ${sol_dir}/[file tail ${sol_dir}]_data.json
set f [open ${sol_data_file} r]
set sol_data [read $f]
close $f
set vhdl_entries ""
foreach v ${vhdl_files} {
  file copy -force vhdl/${v} ${sol_dir}/impl/verilog/${v}
  append vhdl_entries "\n      \"impl\\/verilog\\/${v}\","
}
if {[string first {"Verilog": [} $sol_data] < 0} {
  error "Could not find the Verilog file list in ${sol_data_file} to add the VHDL to"
}
set sol_data [string map [list {"Verilog": [} "\"Verilog\": \[${vhdl_entries}"] $sol_data]
set f [open ${sol_data_file} w]
puts -nonewline $f $sol_data
close $f
set cwd [pwd]
cd ${sol_dir}/impl/ip
exec sh pack.sh > ${cwd}/vhdl_ippack.log 2>@1
cd $cwd
quit
