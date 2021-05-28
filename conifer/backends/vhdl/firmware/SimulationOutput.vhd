-- #########################################################################
-- #########################################################################
-- ###                                                                   ###
-- ###   Use of this code, whether in its current form or modified,      ###
-- ###   implies that you consent to the terms and conditions, namely:   ###
-- ###    - You acknowledge my contribution                              ###
-- ###    - This copyright notification remains intact                   ###
-- ###                                                                   ###
-- ###   Many thanks,                                                    ###
-- ###     Dr. Andrew W. Rose, Imperial College London, 2018             ###
-- ###                                                                   ###
-- #########################################################################
-- #########################################################################

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
use std.textio.all;

library BDT;
use BDT.Types.all;
use BDT.Constants.all;

entity SimulationOutput is
  generic(
    FileName : string := "SimulationOutput.txt";
    FilePath : string := "./"
  );
  port(
    clk    : in std_logic;
    y : in tyArray(nClasses - 1 downto 0) := (others => to_ty(0));
    v : in boolean := false
  );
end SimulationOutput;
-- -------------------------------------------------------------------------
-- -------------------------------------------------------------------------
architecture rtl of SimulationOutput is
  signal v_last : boolean := false;
begin
-- pragma synthesis_off
  process(clk)
    file f     : text open write_mode is FilePath & FileName;
    variable s : line;
  begin
  if rising_edge(clk) then
    if v then
      for i in  y'range loop
        write(s, to_integer(y(i)), right, 10);
        write(s, string'(" "), right, 1);
      end loop;
      writeline( f , s );
    end if;
    v_last <= v;
    if v_last and not v then
      std.env.finish;
    end if;
  end if;
  end process;
-- pragma synthesis_on    
end architecture rtl;
