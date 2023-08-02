#################
#    HLS4ML
#################
set tcldir [file dirname [info script]]
source [file join $tcldir hls_parameters.tcl]

array set opt {
    reset      0
    csim       1
    synth      1
    cosim      0
    export     0
    vsynth     0
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\w+)" $arg unused opt($o)
  }
}

file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results.log"

if {$opt(reset)} {
    open_project -reset ${prj_name}
} else {
    open_project ${prj_name}
}

set_top ${top}
add_files firmware/BDT.h -cflags "-std=c++0x"
add_files firmware/BDT.cpp -cflags "-std=c++0x"
add_files firmware/${prj_name}.cpp -cflags "-std=c++0x"
add_files -tb ${prj_name}_test.cpp -cflags "-I firmware/ -std=c++0x"
add_files -tb tb_data
if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}

open_solution -reset "solution1" -flow_target ${flow_target}
set_part ${part}
create_clock -period 5 -name default

config_interface -m_axi_addr64=${m_axi_addr64}

if {$opt(csim)} {
    csim_design
}

if {$opt(synth)} {
    csynth_design
}

if {$opt(cosim)} {
    cosim_design -trace_level all
}

if {$opt(export)} {
    export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
}

if {$opt(vsynth)} {
    puts "NOT IMPLEMENTED YET"
}
exit
