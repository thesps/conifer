library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;
use work.Arrays0;

entity BDTTop is
  port(
    clk : in std_logic;  -- clock
    X : in txArray(0 to nFeatures-1) := (others => to_tx(0));           -- input features
    X_vld : in boolean := false; -- input valid
    y : out tyArray(0 to nClasses-1) := (others => to_ty(0));            -- output score
    y_vld : out boolArray(0 to nClasses-1) := (others => false) -- output valid
  );
end BDTTop;

architecture rtl of BDTTop is
begin

  BDT0 : entity work.BDT
  generic map(
    iFeature => Arrays0.feature,
    iChildLeft => Arrays0.children_left,
    iChildRight => Arrays0.children_right,
    iParent => Arrays0.parent,
    iLeaf => Arrays0.iLeaf,
    depth => Arrays0.depth,
    threshold => Arrays0.threshold,
    value => Arrays0.value,
    initPredict => Arrays0.initPredict
  )
  port map(
    clk => clk,
    X => X,
    X_vld => X_vld,
    y => y(0),
    y_vld => y_vld(0)
  );

  
end rtl;
