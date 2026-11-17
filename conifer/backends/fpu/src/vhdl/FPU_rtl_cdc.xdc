# Copyright CERN 2026.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-P v2
# Source location: https://github.com/thesps/conifer

# The clock domain crossings of FPU_rtl's configuration port (see FPU_rtl.vhd). The held access and reply, and their
# toggles, cross to the cfg2fpu_*_s and fpu2cfg_*_s registers, which sample them at least two clocks after the toggle.
# So their delay is limited to the FPU clock period, which suffices if the configuration clock is no faster.
# For use with the clocks of the design, which are not otherwise related by this file.
# conifer insert max_delay
set_max_delay -datapath_only -from [get_cells -hier -filter {NAME =~ *cfg2fpu_*_h_reg*}] -to [get_cells -hier -filter {NAME =~ *cfg2fpu_*_s_reg*}] $fpu_rtl_cdc_max_delay
set_max_delay -datapath_only -from [get_cells -hier -filter {NAME =~ *fpu2cfg_*_h_reg*}] -to [get_cells -hier -filter {NAME =~ *fpu2cfg_*_s_reg*}] $fpu_rtl_cdc_max_delay
