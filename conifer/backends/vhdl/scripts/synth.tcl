add_files firmware/
set_property file_type {VHDL 2008} [get_files]
# hls4ml insert synth_design
report_utilization -file util.rpt
