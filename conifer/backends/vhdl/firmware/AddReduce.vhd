library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

entity AddReduce is
generic(
  id : string := "0"
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
  id : string := "0"
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

G1 : if d'length <= 1 generate
    q <= d;
end generate;

G2 : if d'length = 2 generate
    process(clk)
    begin
        if rising_edge(clk) then
            q(q'left) <= d(d'left) + d(d'right);
            q_vld(q'left) <= d_vld(d'left) and d_vld(d'right);
        end if;
    end process;
end generate;

GN : if d'length > 2 generate
  -- Lengths are rounded up to nearest even
  signal dInt : tyArray(0 to intLen - 1) := (others => (others => '0'));
  signal qInt : tyArray(0 to qLen - 1) := (others => (others => '0'));
  signal d_vldInt : boolArray(0 to intLen - 1) := (others => false);
  signal q_vldInt : boolArray(0 to qLen - 1) := (others => false);
  begin
    dInt(0 to d'length - 1) <= d;
    d_vldInt(0 to d'length - 1) <= d_vld;
    d_vldInt(d'length to intLen - 1) <= (others => true);

    GNSums:
    for i in 0 to qLen - 1 generate
        Sum:
        process(clk)
        begin
        if rising_edge(clk) then
            qInt(i) <= dInt(2*i) + dInt(2*i+1);
            q_vldInt(i) <= d_vldInt(2*i) and d_vldInt(2*i+1);
        end if;
        end process;
    end generate;

    Reduce : AddReduce
    generic map(id => id & "_C")
    port map(clk, qInt, q_vldInt, q, q_vld);

end generate;

end behavioral;
