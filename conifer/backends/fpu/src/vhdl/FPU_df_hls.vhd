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

-- FPU_df with the HLS ap_ctrl_chain block protocol, for use as an HLS RTL blackbox (see FPU_df_bb in fpu.h)
-- ap_start presents X, ap_done presents y until ap_continue. One sample is processed per call.
-- ap_ce is not used: HLS holds it high.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fpu_pkg.all;

entity FPU_df_hls is
  port(
    ap_clk      : in  std_logic;
    ap_rst      : in  std_logic;
    ap_ce       : in  std_logic;
    ap_start    : in  std_logic;
    ap_done     : out std_logic;
    ap_idle     : out std_logic;
    ap_ready    : out std_logic;
    ap_continue : in  std_logic;
    -- flat vectors, to connect to the Verilog: X[i], and Tree Engine i's memory ports, are the ith field of each
    X           : in  std_logic_vector(NFEATURES * TBITS - 1 downto 0);
    y           : out std_logic_vector(31 downto 0);
    y_ap_vld    : out std_logic;
    nodes_addr  : out std_logic_vector(NTE * NAWIDTH - 1 downto 0);
    nodes_en    : out std_logic_vector(NTE - 1 downto 0);
    nodes_q     : in  std_logic_vector(NTE * NODEBITS - 1 downto 0);
    roots_addr  : out std_logic_vector(NTE * RAWIDTH - 1 downto 0);
    roots_en    : out std_logic_vector(NTE - 1 downto 0);
    roots_q     : in  std_logic_vector(NTE * ADDRBITS - 1 downto 0)
  );
end entity;

architecture rtl of FPU_df_hls is
  signal X_rdy   : std_logic;
  signal X_vld   : std_logic;
  signal X_a     : features_t;
  signal y_df    : ty;
  signal na      : nodes_addr_array(0 to NTE - 1);
  signal ne      : std_logic_vector(0 to NTE - 1);
  signal nq      : nodes_word_array(0 to NTE - 1);
  signal ra      : roots_addr_array(0 to NTE - 1);
  signal re      : std_logic_vector(0 to NTE - 1);
  signal rq      : roots_word_array(0 to NTE - 1);
  signal y_vld   : std_logic;
  signal running : std_logic := '0'; -- a sample was accepted and its y not yet handed over
  signal done_r  : std_logic := '0';
  signal y_r     : std_logic_vector(31 downto 0) := (others => '0');
begin

  X_vld    <= ap_start and X_rdy and not running;
  ap_ready <= X_vld;
  ap_idle  <= not running;
  ap_done  <= done_r;
  y_ap_vld <= done_r;
  y        <= y_r;

  process(ap_clk)
  begin
    if rising_edge(ap_clk) then
      if ap_rst = '1' then
        running <= '0';
        done_r  <= '0';
      else
        if X_vld = '1' then
          running <= '1';
        end if;
        if y_vld = '1' then
          -- sign extend the score to the 32 bit interface
          y_r    <= std_logic_vector(resize(y_df, 32));
          done_r <= '1';
        end if;
        if done_r = '1' and ap_continue = '1' then
          done_r  <= '0';
          running <= '0';
        end if;
      end if;
    end if;
  end process;

  -- between the flat vectors and the arrays
  Xs : for i in 0 to NFEATURES - 1 generate
    X_a(i) <= signed(X((i + 1) * TBITS - 1 downto i * TBITS));
  end generate;
  Mems : for i in 0 to NTE - 1 generate
    nodes_addr((i + 1) * NAWIDTH - 1 downto i * NAWIDTH) <= na(i);
    nodes_en(i) <= ne(i);
    nq(i) <= nodes_q((i + 1) * NODEBITS - 1 downto i * NODEBITS);
    roots_addr((i + 1) * RAWIDTH - 1 downto i * RAWIDTH) <= ra(i);
    roots_en(i) <= re(i);
    rq(i) <= roots_q((i + 1) * ADDRBITS - 1 downto i * ADDRBITS);
  end generate;

  DF : entity work.FPU_df
    port map(
      clk => ap_clk, rst => ap_rst, X => X_a, X_vld => X_vld, X_rdy => X_rdy, y => y_df, y_vld => y_vld,
      nodes_addr => na, nodes_en => ne, nodes_q => nq,
      roots_addr => ra, roots_en => re, roots_q => rq
    );

end architecture;
