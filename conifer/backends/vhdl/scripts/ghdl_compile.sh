ghdl -a --std=08 --work=BDT ./firmware/Constants.vhd
ghdl -a --std=08 --work=BDT ./firmware/Types.vhd
ghdl -a --std=08 --work=BDT ./firmware/Tree.vhd
ghdl -a --std=08 --work=BDT ./firmware/AddReduce.vhd

# insert arrays

ghdl -a --std=08 --work=BDT ./firmware/BDT.vhd
ghdl -a --std=08 --work=BDT ./firmware/BDTTop.vhd

ghdl -a --std=08 --work=xil_defaultlib ./firmware/SimulationInput.vhd
ghdl -a --std=08 --work=xil_defaultlib ./firmware/SimulationOutput.vhd
ghdl -a --std=08 --work=xil_defaultlib ./firmware/BDTTestbench.vhd

ghdl -e --std=08 --work=xil_defaultlib testbench