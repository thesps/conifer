library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;

entity Split_lte is
  port(
    a : in tx;
    b : in tx;
    q : out boolean
  );
end Split_lte;

architecture rtl of Split_lte is
begin
  q <= a <= b;
end rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;

entity Split_lt is
  port(
    a : in tx;
    b : in tx;
    q : out boolean
  );
end Split_lt;

architecture rtl of Split_lt is
begin
  q <= a < b;
end rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;

entity Split is
  port(
    a : in tx;
    b : in tx;
    q : out boolean
  );
end Split;

architecture rtl of Split is
begin
  -- conifer insert split
end rtl;