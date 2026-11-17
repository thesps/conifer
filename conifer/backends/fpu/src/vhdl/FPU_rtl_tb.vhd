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

-- Testbench for FPU_rtl, with asynchronous FPU and configuration clocks. Reads the testbench data written by
-- FPUBuilder.write_testbench_data for an FPU without the dynamic scaler:
--   CFG_FILE : the configuration writes loading the model, one per line: address and data, in hexadecimal
--   X_FILE   : the number of samples and the number of features, then one sample per line
-- Loads the model with the writes of CFG_FILE, then reads back each address written, writing the data read to
-- READBACK_FILE, and the INFO words to INFO_FILE, one per line in hexadecimal. Then runs the samples, writing one y
-- per line to Y_FILE. Reports configuration errors.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fpu_pkg.all;
use work.fpu_rtl_pkg.all;

entity FPU_rtl_tb is
  generic(
    CFG_FILE      : string := "../tb_data/cfg_writes.dat";
    X_FILE        : string := "../tb_data/X.dat";
    READBACK_FILE : string := "readback.dat";
    INFO_FILE     : string := "info.dat";
    Y_FILE        : string := "y.dat"
  );
end entity;

architecture sim of FPU_rtl_tb is
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

  constant X_mem : x_mem_t := read_X;

  signal clk       : std_logic := '0';
  signal cfg_clk   : std_logic := '0';
  signal rst       : std_logic := '1';
  signal cfg_rst   : std_logic := '1';
  signal X         : features_t := (others => (others => '0'));
  signal X_vld     : std_logic := '0';
  signal X_rdy     : std_logic;
  signal y         : ty;
  signal y_vld     : std_logic;
  signal cfg_req   : std_logic := '0';
  signal cfg_we    : std_logic := '0';
  signal cfg_addr  : cfg_addr_t := (others => '0');
  signal cfg_wdata : cfg_data_t := (others => '0');
  signal cfg_busy  : std_logic;
  signal cfg_ack   : std_logic;
  signal cfg_rdata : cfg_data_t;
  signal cfg_err   : std_logic;
  signal loaded    : boolean := false;
  signal finished  : boolean := false;
begin

  -- asynchronous clocks, the configuration clock slower, like an ipbus clock
  clk     <= not clk after 1.667 ns when not finished;
  cfg_clk <= not cfg_clk after 16.03 ns when not finished;

  -- load the model, then read it back
  process
    file fc : text open read_mode is CFG_FILE;
    file fr : text open write_mode is READBACK_FILE;
    file fi : text open write_mode is INFO_FILE;
    variable l, lo : line;
    variable a, d : std_logic_vector(31 downto 0);
    variable n_err : integer := 0;
    variable n_writes : integer := 0;

    procedure cfg_access(we : std_logic; addr : std_logic_vector; data : std_logic_vector) is
    begin
      cfg_we    <= we;
      cfg_addr  <= addr(CFG_ABITS - 1 downto 0);
      cfg_wdata <= data;
      cfg_req   <= '1';
      wait until rising_edge(cfg_clk);
      cfg_req   <= '0';
      wait until rising_edge(cfg_clk) and cfg_ack = '1';
      if cfg_err = '1' then
        n_err := n_err + 1;
      end if;
    end procedure;
  begin
    wait for 100 ns;
    wait until rising_edge(cfg_clk);
    cfg_rst <= '0';
    while not endfile(fc) loop
      readline(fc, l);
      hread(l, a);
      hread(l, d);
      cfg_access('1', a, d);
      n_writes := n_writes + 1;
    end loop;
    file_close(fc);
    file_open(fc, CFG_FILE, read_mode);
    while not endfile(fc) loop
      readline(fc, l);
      hread(l, a);
      cfg_access('0', a, x"00000000");
      hwrite(lo, cfg_rdata);
      writeline(fr, lo);
    end loop;
    for i in 0 to N_INFO - 1 loop
      cfg_access('0', std_logic_vector(to_unsigned(REGION_INFO * 2**REGION_LO + i, 32)), x"00000000");
      hwrite(lo, cfg_rdata);
      writeline(fi, lo);
    end loop;
    report "FPU_rtl_tb: " & integer'image(n_writes) & " configuration writes, " & integer'image(n_err) & " errors";
    loaded <= true;
    wait;
  end process;

  -- then send the samples
  process
  begin
    wait for 50 ns;
    wait until rising_edge(clk);
    rst <= '0';
    wait until loaded;
    for s in 0 to NSAMPLES - 1 loop
      X     <= X_mem(s);
      X_vld <= '1';
      wait until rising_edge(clk) and X_rdy = '1';
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
    report "FPU_rtl_tb: " & integer'image(NSAMPLES) & " samples";
    finished <= true;
    wait;
  end process;

  UUT : entity work.FPU_rtl
    port map(
      clk => clk, rst => rst, X => X, X_vld => X_vld, X_rdy => X_rdy, y => y, y_vld => y_vld,
      cfg_clk => cfg_clk, cfg_rst => cfg_rst, cfg_req => cfg_req, cfg_we => cfg_we, cfg_addr => cfg_addr,
      cfg_wdata => cfg_wdata, cfg_busy => cfg_busy, cfg_ack => cfg_ack, cfg_rdata => cfg_rdata, cfg_err => cfg_err
    );

end architecture;
