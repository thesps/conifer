-- Copyright CERN 2026.
--
-- This source describes Open Hardware and is licensed under the CERN-OHL-P v2
-- You may redistribute and modify this documentation and make products
-- using it under the terms of the CERN-OHL-P v2 (https:/cern.ch/cern-ohl).
--
-- This code is distributed WITHOUT ANY EXPRESS OR IMPLIED
-- WARRANTY, INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY
-- AND FITNESS FOR A PARTICULAR PURPOSE. Please see the CERN-OHL-P v2
-- for applicable conditions
--
-- Source location: https://github.com/thesps/conifer

-- A Tree Engine memory (nodes or roots), one clock, read latency 1 on both ports:
--   port a: read only, by the Tree Engine
--   port b: read and write, by the configuration port. Read first: a read with a write returns the old word.
--           Write only if not B_READ: a block RAM is then simple dual port, which is up to twice as wide as true
--           dual port, so takes up to half the block RAMs for words wider than 36 bits. b_q is then all zero
-- STYLE is the Vivado ram_style: "block", "distributed" or "ultra". Initially all zero.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity FPU_mem is
  generic(
    WIDTH     : integer;
    ABITS     : integer;
    STYLE     : string := "block";
    B_READ    : boolean := true
  );
  port(
    clk    : in  std_logic;
    a_en   : in  std_logic;
    a_addr : in  std_logic_vector(ABITS - 1 downto 0);
    a_q    : out std_logic_vector(WIDTH - 1 downto 0);
    b_en   : in  std_logic;
    b_we   : in  std_logic;
    b_addr : in  std_logic_vector(ABITS - 1 downto 0);
    b_d    : in  std_logic_vector(WIDTH - 1 downto 0);
    b_q    : out std_logic_vector(WIDTH - 1 downto 0)
  );
end entity;

architecture rtl of FPU_mem is
  type mem_t is array(0 to 2**ABITS - 1) of std_logic_vector(WIDTH - 1 downto 0);
  signal mem : mem_t := (others => (others => '0'));
  attribute ram_style : string;
  attribute ram_style of mem : signal is STYLE;
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if a_en = '1' then
        a_q <= mem(to_integer(unsigned(a_addr)));
      end if;
      if b_en = '1' then
        if B_READ then
          b_q <= mem(to_integer(unsigned(b_addr)));
        else
          b_q <= (others => '0');
        end if;
        if b_we = '1' then
          mem(to_integer(unsigned(b_addr))) <= b_d;
        end if;
      end if;
    end if;
  end process;
end architecture;
