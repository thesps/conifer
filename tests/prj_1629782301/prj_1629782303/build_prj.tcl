#################
#    HLS4ML
#################
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
    open_project -reset my_prj_prj
} else {
    open_project my_prj_prj
}

set_top my_prj
add_files firmware/BDT.h -cflags "-std=c++0x"
add_files firmware/my_prj.cpp -cflags "-std=c++0x"
add_files -tb my_prj_test.cpp -cflags "-I firmware/ -std=c++0x"
add_files -tb tb_data
if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}

open_solution -reset "solution1"
set_part {XAZU5EV-figd2104-2L-e}
create_clock -period 5 -name default

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
    export_design -format ip_catalog
}

if {$opt(vsynth)} {
    puts "NOT IMPLEMENTED YET"
}
exit
