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
source [file join $tcldir accelerator_parameters.tcl]

create_project project_1 ${prj_name}_vivado -part ${part} -force

set_property board_part ${board_part} [current_project]
set_property  ip_repo_paths  ${prj_name} [current_project]
update_ip_catalog

create_bd_design "design_1"

startgroup
create_bd_cell -type ip -vlnv ${processing_system_ip} processing_system_0
endgroup

apply_bd_automation -rule xilinx.com:bd_rule:${processing_system} -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system_0]

startgroup
set_property -dict [list ${ps_config}] [get_bd_cells processing_system_0]
endgroup

startgroup
create_bd_cell -type ip -vlnv cern.ch:conifer:${top}:${version} ${ip_name}
endgroup

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system_0/${ps_m_axi_port}} Slave {/${ip_name}/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins ${ip_name}/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/${ip_name}/m_axi_gmem0} Slave {/processing_system_0/${ps_s_axi_port}} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins processing_system_0/${ps_s_axi_port}]
endgroup

make_wrapper -files [get_files ./${prj_name}_vivado/project_1.srcs/sources_1/bd/design_1/design_1.bd] -top

add_files -norecurse ./${prj_name}_vivado/project_1.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run -timeout 360 impl_1

write_hw_platform -fixed -include_bit -force -file ./${prj_name}_vivado/${prj_name}.xsa

open_run impl_1
report_utilization -file util.rpt -hierarchical -hierarchical_percentages