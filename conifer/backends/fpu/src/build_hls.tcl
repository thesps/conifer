set tcldir [file dirname [info script]]
source [file join $tcldir parameters.tcl]

open_project -reset ${prj_name}

set_top ${top}

add_files fpu.cpp -cflags "-std=c++17"

open_solution -reset "solution1" -flow_target ${flow_target}

set_part ${part}
create_clock -period ${clock_period} -name clk

config_interface -m_axi_addr64=${m_axi_addr64}

csynth_design
export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
quit
