library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

entity AddReduce is
generic(
  id : string := "0";
  depth : integer := 0;
  pipelining : integer := 1
);
port(
  clk   : in std_logic := '0';
  d     : in tyArray;
  d_vld : in boolArray;
  q     : out tyArray;
  q_vld : out boolArray
);
end AddReduce;

architecture behavioral of AddReduce is

constant len : integer := d'length;
constant intLen : integer := 2 * ((len + 1) / 2);
constant qLen : integer := (len + 1) / 2;

component AddReduce is
generic(
  id : string := "0";
  depth : integer := 0;
  pipelining : integer := 1
);
port(
  clk   : in std_logic := '0';
  d     : in tyArray; --(0 to intLen / 2 - 1);
  d_vld : in boolArray;
  q     : out tyArray; --(0 to qLen / 2 - 1) 
  q_vld : out boolArray
);
end component AddReduce;

begin

GN : if d'length <= 1 generate
    q <= d;
    q_vld <= d_vld;
else generate
  -- Lengths are rounded up to nearest even
  signal dInt : tyArray(0 to intLen - 1) := (others => (others => '0'));
  signal qInt : tyArray(0 to qLen - 1) := (others => (others => '0'));
  signal qReg : tyArray(0 to qLen - 1) := (others => (others => '0'));
  signal d_vldInt : boolArray(0 to intLen - 1) := (others => false);
  signal q_vldInt : boolArray(0 to qLen - 1) := (others => false);
  signal q_vldReg : boolArray(0 to qLen - 1) := (others => false);
  begin
    dInt(0 to d'length - 1) <= d;
    d_vldInt(0 to d'length - 1) <= d_vld;
    d_vldInt(d'length to intLen - 1) <= (others => true);

    GNSums:
    for i in 0 to qLen - 1 generate
      qInt(i) <= dInt(2*i) + dInt(2*i+1);
      q_vldInt(i) <= d_vldInt(2*i) and d_vldInt(2*i+1);
    end generate;

    GRegister:
    -- insert a register every "pipelining" stages
    if depth mod pipelining = 0 generate
      process(clk)
      begin
        if rising_edge(clk) then
          qReg <= qInt;
          q_vldReg <= q_vldInt;
        end if;
      end process;
    else generate
      qReg <= qInt;
      q_vldReg <= q_vldInt;
    end generate;

    Reduce : AddReduce
    generic map(id & "_C", depth+1, pipelining)
    port map(clk, qReg, q_vldReg, q, q_vld);

end generate;

end behavioral;
