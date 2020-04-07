library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;
-- include arrays

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

-- instantiate BDTs

end rtl;
