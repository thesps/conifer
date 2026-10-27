library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

-- Register wrapper of BDTTop for out-of-context implementation
-- so that every timed path through BDTTop starts and ends at a register
entity BDTTopOOC is
  port(
    clk : in std_logic;  -- clock
    X : in txArray(0 to nFeatures-1) := (others => to_tx(0));           -- input features
    X_vld : in boolean := false; -- input valid
    y : out tyArray(0 to nClasses-1) := (others => to_ty(0));            -- output score
    y_vld : out boolArray(0 to nClasses-1) := (others => false) -- output valid
  );
end BDTTopOOC;

architecture rtl of BDTTopOOC is
  signal X_reg : txArray(0 to nFeatures-1) := (others => to_tx(0));
  signal X_vld_reg : boolean := false;
  signal y_int : tyArray(0 to nClasses-1) := (others => to_ty(0));
  signal y_vld_int : boolArray(0 to nClasses-1) := (others => false);
begin

  X_reg <= X when rising_edge(clk);
  X_vld_reg <= X_vld when rising_edge(clk);

  BDTTop_inst : entity work.BDTTop
  port map(
    clk => clk,
    X => X_reg,
    X_vld => X_vld_reg,
    y => y_int,
    y_vld => y_vld_int
  );

  y <= y_int when rising_edge(clk);
  y_vld <= y_vld_int when rising_edge(clk);

end rtl;
