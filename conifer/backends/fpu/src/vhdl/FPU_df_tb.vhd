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
-- Reads the testbench data written by FPUBuilder.write_testbench_data, for an FPU without the dynamic scaler:
--   NODES_FILE : NNODES lines per Tree Engine of the 7 node fields: threshold score feature child_left child_right iclass is_leaf
--   ROOTS_FILE : one line per Tree Engine: the number of trees, then the root node addresses (NROOTS + 1 values)
--   X_FILE     : the number of samples and the number of features, then one sample per line
-- Writes one y per line to Y_FILE, and reports the cycles from the first X accepted to the last y, and the node memory reads
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fpu_pkg.all;

entity FPU_df_tb is
  generic(
    NODES_FILE : string := "../tb_data/nodes.dat";
    ROOTS_FILE : string := "../tb_data/roots.dat";
    X_FILE     : string := "../tb_data/X.dat";
    Y_FILE     : string := "y.dat"
  );
end entity;

architecture sim of FPU_df_tb is
  constant NDEPTH   : integer := 2**NAWIDTH;
  constant RDEPTH   : integer := 2**RAWIDTH;
  type node_mem_t is array(0 to NTE * NDEPTH - 1) of node_word_t;
  type root_mem_t is array(0 to NTE * RDEPTH - 1) of roots_word_t;

  -- the node memory word from the 7 node fields, as written to the nodes file
  function to_node_word(v : integer_vector(0 to 6)) return node_word_t is
    variable n : node_t;
  begin
    n.threshold   := to_signed(v(0), TBITS);
    n.score       := to_signed(v(1), UBITS);
    n.feature     := unsigned(to_signed(v(2), FEATBITS));
    n.child_left  := to_signed(v(3), ADDRBITS);
    n.child_right := to_signed(v(4), ADDRBITS);
    n.iclass      := to_unsigned(v(5), CLASSBITS);
    n.is_leaf     := '1' when v(6) = 1 else '0';
    return to_std_logic_vector(n);
  end function;

  impure function read_nodes return node_mem_t is
    file f : text open read_mode is NODES_FILE;
    variable l : line;
    variable v : integer_vector(0 to 6);
    -- addresses beyond NNODES hold null nodes
    variable m : node_mem_t := (others => to_node_word((0, 0, -2, -1, -1, 0, 1)));
  begin
    for te in 0 to NTE - 1 loop
      for i in 0 to NNODES - 1 loop
        readline(f, l);
        for j in v'range loop
          read(l, v(j));
        end loop;
        m(te * NDEPTH + i) := to_node_word(v);
      end loop;
    end loop;
    return m;
  end function;

  impure function read_roots return root_mem_t is
    file f : text open read_mode is ROOTS_FILE;
    variable l : line;
    variable v : integer;
    variable m : root_mem_t := (others => (others => '0'));
  begin
    for te in 0 to NTE - 1 loop
      readline(f, l);
      for i in 0 to NROOTS loop
        read(l, v);
        m(te * RDEPTH + i) := std_logic_vector(to_signed(v, ADDRBITS));
      end loop;
    end loop;
    return m;
  end function;

  -- the number of samples, or of features, from the first line of X_FILE
  impure function read_X_shape(dim : integer) return integer is
    file f : text open read_mode is X_FILE;
    variable l : line;
    variable v : integer_vector(0 to 1);
  begin
    readline(f, l);
    read(l, v(0));
    read(l, v(1));
    return v(dim);
  end function;

  constant NSAMPLES : integer := read_X_shape(0);
  type x_mem_t is array(0 to NSAMPLES - 1) of features_t;

  impure function read_X return x_mem_t is
    file f : text open read_mode is X_FILE;
    variable l : line;
    variable v : integer;
    variable m : x_mem_t := (others => (others => (others => '0')));
  begin
    readline(f, l);
    for i in m'range loop
      readline(f, l);
      for j in 0 to read_X_shape(1) - 1 loop
        read(l, v);
        m(i)(j) := to_signed(v, TBITS);
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
    report "FPU_df_tb: " & integer'image(NSAMPLES) & " samples, " & integer'image(cycle - t_first) &
           " cycles from the first X accepted to the last y, " & integer'image(n_reads) & " node memory reads";
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
