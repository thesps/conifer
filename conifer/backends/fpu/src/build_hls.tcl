set tcldir [file dirname [info script]]
source [file join $tcldir parameters.tcl]

open_project -reset ${prj_name}

set_top FPU

add_files fpu.cpp -cflags "-std=c++17"

open_solution -reset "solution1"

set_part ${part}
create_clock -period 10 -name clk

config_interface -m_axi_addr64=false

csynth_design
export_design
quit
