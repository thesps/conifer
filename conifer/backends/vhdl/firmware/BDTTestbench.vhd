library ieee;
use ieee.std_logic_1164.all;

library BDT;
use BDT.Constants.all;
use BDT.Types.all;

entity testbench is
end testbench;

architecture rtl of testbench is
  signal X : txArray(0 to nFeatures - 1) := (others => to_tx(0));
  signal X_vld : boolean := false;
  signal y : tyArray(0 to nClasses - 1) := (others => to_ty(0));
  signal y_vld : boolArray(0 to nClasses - 1) := (others => false);
  signal clk : std_logic := '0';
begin
    clk <= not clk after 2.5 ns;

    Input : entity work.SimulationInput
    port map(clk, X, X_vld);

    UUT : entity BDT.BDTTop
    port map(clk, X, X_vld, y, y_vld);

    Output : entity work.SimulationOutput
    port map(clk, y, y_vld(0));

end architecture rtl;
