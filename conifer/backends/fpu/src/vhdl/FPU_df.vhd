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

-- Forest Processing Unit decision function
-- Runs NTE Tree Engines on one sample X and sums their scores in a pipelined adder tree.
-- X is accepted when X_vld = '1' and X_rdy = '1'. y is valid for the one cycle that y_vld = '1'.
-- A new X is accepted as soon as the Tree Engines finish, while the previous sample's sum is still in the adder tree.
-- The node and roots memories of each Tree Engine are external, with one cycle read latency.
-- Element i of each memory port array is the port of Tree Engine i. The configuration is in fpu_pkg.
--
-- The Tree Engines need not finish together: each holds its y, and done, from finishing until the next start.
-- The adder tree takes all the Tree Engines' y once the last one is done, so a sample takes as long as its slowest
-- Tree Engine. The Tree Engines then start the next sample together.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fpu_pkg.all;

entity FPU_df is
  port(
    clk        : in  std_logic;
    rst        : in  std_logic;
    X          : in  features_t;
    X_vld      : in  std_logic;
    X_rdy      : out std_logic;
    y          : out ty;
    y_vld      : out std_logic;
    nodes_addr : out nodes_addr_array(0 to NTE - 1);
    nodes_en   : out std_logic_vector(0 to NTE - 1);
    nodes_q    : in  nodes_word_array(0 to NTE - 1);
    roots_addr : out roots_addr_array(0 to NTE - 1);
    roots_en   : out std_logic_vector(0 to NTE - 1);
    roots_q    : in  roots_word_array(0 to NTE - 1)
  );
end entity;

architecture rtl of FPU_df is
  constant NLEV     : integer := clog2(NTE);   -- adder tree levels
  constant NPAD     : integer := 2**NLEV;      -- Tree Engines padded to a power of 2

  signal X_r       : features_t := (others => (others => '0'));
  signal busy      : std_logic := '0';
  signal start     : std_logic;
  signal te_done   : std_logic_vector(0 to NTE - 1);
  signal te_y      : ty_array(0 to NTE - 1);
  signal all_done  : std_logic;

  subtype level_t is ty_array(0 to NPAD - 1);
  type sum_t is array(0 to NLEV) of level_t;
  signal sums      : sum_t := (others => (others => (others => '0')));
  signal sums_vld  : std_logic_vector(0 to NLEV) := (others => '0');
begin

  X_rdy <= not busy;
  start <= X_vld and not busy;
  all_done <= '1' when te_done = (te_done'range => '1') else '0';

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        busy <= '0';
      elsif start = '1' then
        X_r  <= X;
        busy <= '1';
      elsif busy = '1' and all_done = '1' then
        -- the Tree Engines clear done when started, so all_done is only seen once they finish
        busy <= '0';
      end if;
    end if;
  end process;

  TEs : for i in 0 to NTE - 1 generate
    TE : entity work.TreeEngine
      port map(
        clk        => clk,
        rst        => rst,
        start      => start,
        -- X is registered on start, before the Tree Engines first use it
        X          => X_r,
        y          => te_y(i),
        done       => te_done(i),
        nodes_addr => nodes_addr(i),
        nodes_en   => nodes_en(i),
        nodes_q    => nodes_q(i),
        roots_addr => roots_addr(i),
        roots_en   => roots_en(i),
        roots_q    => roots_q(i)
      );
  end generate;

  -- pipelined adder tree, one register per level. The scores wrap on overflow, like the HLS score type
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        sums_vld <= (others => '0');
      else
        sums_vld(0) <= busy and all_done;
        for l in 1 to NLEV loop
          sums_vld(l) <= sums_vld(l - 1);
        end loop;
      end if;
      for i in 0 to NPAD - 1 loop
        if i < NTE then
          sums(0)(i) <= te_y(i);
        else
          sums(0)(i) <= (others => '0');
        end if;
      end loop;
      for l in 1 to NLEV loop
        for i in 0 to NPAD / 2**l - 1 loop
          sums(l)(i) <= sums(l - 1)(2 * i) + sums(l - 1)(2 * i + 1);
        end loop;
      end loop;
    end if;
  end process;

  y     <= sums(NLEV)(0);
  y_vld <= sums_vld(NLEV);

end architecture;
