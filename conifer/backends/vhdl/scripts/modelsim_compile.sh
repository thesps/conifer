mkdir msimbdtlib
vlib msimbdtlib/BDT
vmap BDT msimbdtlib/BDT

vcom -2008 -work BDT ./firmware/Constants.vhd
vcom -2008 -work BDT ./firmware/Types.vhd
vcom -2008 -work BDT ./firmware/Tree.vhd
vcom -2008 -work BDT ./firmware/AddReduce.vhd
# insert arrays
vcom -2008 -work BDT ./firmware/BDT.vhd
vcom -2008 -work BDT ./firmware/BDTTop.vhd

vlib msimbdtlib/xil_defaultlib
vmap work msimbdtlib/xil_defaultlib
vcom -2008 -work xil_defaultlib ./firmware/SimulationInput.vhd
vcom -2008 -work xil_defaultlib ./firmware/SimulationOutput.vhd
vcom -2008 -work xil_defaultlib ./firmware/BDTTestbench.vhd
