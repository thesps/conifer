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

-- Standalone testbench for FPU_df, with one cycle latency memory models. The configuration is in fpu_pkg.
-- Reads packed node and roots words, and X, as binary strings one per line:
--   NODES_FILE : NTE * 2**NAWIDTH node words, Tree Engine by Tree Engine
--   ROOTS_FILE : NTE * 2**RAWIDTH roots words
--   X_FILE     : one sample per line, X[NFEATURES-1] first
-- Writes one y per line to Y_FILE, then the number of cycles from the first X accepted to the last y
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fpu_pkg.all;

entity FPU_df_tb is
  generic(
    NSAMPLES   : integer := 1;
    NODES_FILE : string := "nodes.mem";
    ROOTS_FILE : string := "roots.mem";
    X_FILE     : string := "X.mem";
    Y_FILE     : string := "y.out"
  );
end entity;

architecture sim of FPU_df_tb is
  constant NDEPTH   : integer := 2**NAWIDTH;
  constant RDEPTH   : integer := 2**RAWIDTH;
  type node_mem_t is array(0 to NTE * NDEPTH - 1) of node_word_t;
  type root_mem_t is array(0 to NTE * RDEPTH - 1) of roots_word_t;
  type x_mem_t is array(0 to NSAMPLES - 1) of features_t;

  impure function read_nodes return node_mem_t is
    file f : text open read_mode is NODES_FILE;
    variable l : line;
    variable b : bit_vector(NODEBITS - 1 downto 0);
    variable m : node_mem_t;
  begin
    for i in m'range loop
      readline(f, l);
      read(l, b);
      m(i) := to_stdlogicvector(b);
    end loop;
    return m;
  end function;

  impure function read_roots return root_mem_t is
    file f : text open read_mode is ROOTS_FILE;
    variable l : line;
    variable b : bit_vector(ADDRBITS - 1 downto 0);
    variable m : root_mem_t;
  begin
    for i in m'range loop
      readline(f, l);
      read(l, b);
      m(i) := to_stdlogicvector(b);
    end loop;
    return m;
  end function;

  impure function read_X return x_mem_t is
    file f : text open read_mode is X_FILE;
    variable l : line;
    variable b : bit_vector(NFEATURES * TBITS - 1 downto 0);
    variable m : x_mem_t;
  begin
    for i in m'range loop
      readline(f, l);
      read(l, b);
      for j in 0 to NFEATURES - 1 loop
        m(i)(j) := signed(to_stdlogicvector(b((j + 1) * TBITS - 1 downto j * TBITS)));
      end loop;
    end loop;
    return m;
  end function;

  constant nodes_mem : node_mem_t := read_nodes;
  constant roots_mem : root_mem_t := read_roots;
  constant X_mem     : x_mem_t := read_X;

  signal clk        : std_logic := '0';
  signal rst        : std_logic := '1';
  signal X          : features_t := (others => (others => '0'));
  signal X_vld      : std_logic := '0';
  signal X_rdy      : std_logic;
  signal y          : ty;
  signal y_vld      : std_logic;
  signal nodes_addr : nodes_addr_array(0 to NTE - 1);
  signal nodes_en   : std_logic_vector(0 to NTE - 1);
  signal nodes_q    : nodes_word_array(0 to NTE - 1) := (others => (others => '0'));
  signal roots_addr : roots_addr_array(0 to NTE - 1);
  signal roots_en   : std_logic_vector(0 to NTE - 1);
  signal roots_q    : roots_word_array(0 to NTE - 1) := (others => (others => '0'));
  signal cycle      : integer := 0;
  signal t_first    : integer := -1;
  signal n_reads    : integer := 0; -- node memory reads
  signal finished   : boolean := false;
begin

  clk <= not clk after 5 ns when not finished;

  process(clk)
    variable n : integer;
  begin
    if rising_edge(clk) then
      cycle <= cycle + 1;
      n := 0;
      for i in 0 to NTE - 1 loop
        if nodes_en(i) = '1' then
          nodes_q(i) <= nodes_mem(i * NDEPTH + to_integer(unsigned(nodes_addr(i))));
          n := n + 1;
        end if;
        if roots_en(i) = '1' then
          roots_q(i) <= roots_mem(i * RDEPTH + to_integer(unsigned(roots_addr(i))));
        end if;
      end loop;
      n_reads <= n_reads + n;
    end if;
  end process;

  -- send the samples
  process
  begin
    rst <= '1';
    wait for 52 ns;
    wait until rising_edge(clk);
    rst <= '0';
    for s in 0 to NSAMPLES - 1 loop
      X     <= X_mem(s);
      X_vld <= '1';
      wait until rising_edge(clk) and X_rdy = '1';
      if s = 0 then
        t_first <= cycle;
      end if;
    end loop;
    X_vld <= '0';
    wait;
  end process;

  -- collect the predictions
  process
    file f : text open write_mode is Y_FILE;
    variable l : line;
  begin
    for s in 0 to NSAMPLES - 1 loop
      wait until rising_edge(clk) and y_vld = '1';
      write(l, to_integer(y));
      writeline(f, l);
    end loop;
    -- cycles from the first X accepted to the last y
    write(l, string'("cycles "));
    write(l, cycle - t_first);
    writeline(f, l);
    write(l, string'("node_reads "));
    write(l, n_reads);
    writeline(f, l);
    finished <= true;
    wait;
  end process;

  UUT : entity work.FPU_df
    port map(
      clk => clk, rst => rst, X => X, X_vld => X_vld, X_rdy => X_rdy, y => y, y_vld => y_vld,
      nodes_addr => nodes_addr, nodes_en => nodes_en, nodes_q => nodes_q,
      roots_addr => roots_addr, roots_en => roots_en, roots_q => roots_q
    );

end architecture;
