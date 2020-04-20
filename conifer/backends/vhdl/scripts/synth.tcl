add_files firmware/
remove_files firmware/SimulationInput.vhd
remove_files firmware/SimulationOutput.vhd
remove_files firmware/BDTTestbench.vhd
remove_files firmware/TestUtil.vhd
set_property file_type {VHDL 2008} [get_files]
# hls4ml insert synth_design
report_utilization -file util.rpt
