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

-- Forest Processing Unit Tree Engine
-- Walks the trees loaded into its node memory, summing the score of the leaf reached in each.
-- The roots memory holds the number of trees at address 0, then the address of each tree's root node.
-- Both memories are external, read with one cycle latency: the data for the address presented with en = '1'
-- in one cycle is on q in the next.
--
-- Each traversal step takes two cycles:
--   SELECT : select the next node address (a child, or the next tree's root at a leaf) and read it
--   COMPARE: the node arrives from the memory: compare X[feature] <= threshold and register the node
-- The Tree Engine walks two trees at once, in two threads, interleaved: in each cycle one thread selects while
-- the other compares, so the node memory is read every cycle. Thread t walks trees t, t + 2, t + 4, ...
-- Each thread reads its next tree's root address from the roots memory ahead of time, in its select cycle,
-- so moving on to the next tree costs nothing. The threads share the roots memory port in the same way.
-- The configuration (the field widths, the memory sizes) is in fpu_pkg.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fpu_pkg.all;

entity TreeEngine is
  port(
    clk        : in  std_logic;
    rst        : in  std_logic;
    -- start walking the trees with X, which must be stable until done
    start      : in  std_logic;
    X          : in  features_t;
    -- y is the sum of the scores, valid while done = '1', which is held until the next start
    y          : out ty;
    done       : out std_logic;
    nodes_addr : out nodes_addr_t;
    nodes_en   : out std_logic;
    nodes_q    : in  node_word_t;
    roots_addr : out roots_addr_t;
    roots_en   : out std_logic;
    roots_q    : in  roots_word_t
  );
end entity;

architecture rtl of TreeEngine is
  -- S_COUNT: the number of trees arrives. S_ROOT0 (S_ROOT1): thread 0's (1's) first root address arrives
  type state_t is (S_IDLE, S_COUNT, S_ROOT0, S_ROOT1, S_RUN);
  signal state  : state_t := S_IDLE;
  signal done_r : std_logic := '0';
  signal ph     : integer range 0 to 1 := 0; -- the thread selecting in this cycle

  -- per thread state
  type root_arr_t  is array(0 to 1) of roots_word_t;
  type count_arr_t is array(0 to 1) of unsigned(ADDRBITS - 1 downto 0);
  type ridx_arr_t  is array(0 to 1) of integer range 0 to NROOTS;
  type node_arr_t  is array(0 to 1) of node_t;
  -- the virtual leaf each thread starts on, scoring 0
  constant LEAF0 : node_t := (threshold => (others => '0'), score => (others => '0'), feature => (others => '0'),
                              child_left => (others => '1'), child_right => (others => '1'), iclass => (others => '0'),
                              is_leaf => '1');
  signal active    : std_logic_vector(0 to 1) := (others => '0'); -- the thread has trees to finish
  signal n_left    : count_arr_t := (others => (others => '0'));     -- trees left, including the current one
  signal ridx      : ridx_arr_t := (others => 0);                     -- roots index of the next root to read
  signal root_pf   : std_logic_vector(0 to 1) := (others => '0');     -- a root read was issued in the last cycle
  signal next_root : root_arr_t := (others => (others => '0'));
  signal acc       : ty_array(0 to 1) := (others => (others => '0'));
  signal inflight  : std_logic_vector(0 to 1) := (others => '0');     -- a node read was issued in the last cycle
  -- the registered node, and its comparison
  signal node      : node_arr_t := (others => LEAF0);
  signal go_left   : std_logic_vector(0 to 1) := (others => '0');

  function ridx_add(r : integer; n : integer) return integer is
  begin
    return imin(r + n, NROOTS);
  end function;

  -- X[feature] <= threshold, for the node n
  function compare(n : node_t; Xv : features_t) return std_logic is
    variable f : integer;
  begin
    -- leaves have feature -2: their comparison is unused, so any feature out of range compares X[0]
    f := to_integer(n.feature);
    if f >= NFEATURES then
      f := 0;
    end if;
    if Xv(f) <= n.threshold then
      return '1';
    end if;
    return '0';
  end function;

begin

  y    <= acc(0) + acc(1);
  done <= done_r;

  -- memory reads
  process(state, start, ph, active, node, go_left, n_left, next_root, ridx)
  begin
    roots_en   <= '0';
    roots_addr <= std_logic_vector(to_unsigned(ridx(ph), RAWIDTH));
    nodes_en   <= '0';
    nodes_addr <= (others => '0');
    case state is
      when S_IDLE =>
        -- the number of trees
        if start = '1' then
          roots_en   <= '1';
          roots_addr <= (others => '0');
        end if;
      when S_COUNT =>
        -- thread 0's first root
        roots_en   <= '1';
        roots_addr <= std_logic_vector(to_unsigned(1, RAWIDTH));
      when S_ROOT0 =>
        -- thread 1's first root
        roots_en   <= '1';
        roots_addr <= std_logic_vector(to_unsigned(imin(2, NROOTS), RAWIDTH));
      when S_RUN =>
        if active(ph) = '1' then
          if node(ph).is_leaf = '0' then
            nodes_en <= '1';
            if go_left(ph) = '1' then
              nodes_addr <= std_logic_vector(node(ph).child_left(NAWIDTH - 1 downto 0));
            else
              nodes_addr <= std_logic_vector(node(ph).child_right(NAWIDTH - 1 downto 0));
            end if;
          elsif n_left(ph) > 1 then
            -- the thread's next tree's root node, and the root after it
            nodes_en   <= '1';
            nodes_addr <= next_root(ph)(NAWIDTH - 1 downto 0);
            roots_en   <= '1';
          end if;
        end if;
      when others =>
    end case;
  end process;

  process(clk)
    variable count : unsigned(ADDRBITS - 1 downto 0);
    variable still : std_logic;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state    <= S_IDLE;
        done_r   <= '0';
        active   <= (others => '0');
        root_pf  <= (others => '0');
        inflight <= (others => '0');
      else
        case state is
          when S_IDLE =>
            if start = '1' then
              acc    <= (others => (others => '0'));
              done_r <= '0';
              state  <= S_COUNT;
            end if;

          when S_COUNT =>
            -- share the trees between the threads, each starting on a virtual leaf scoring 0, before its first tree
            count := unsigned(roots_q);
            for t in 0 to 1 loop
              node(t)    <= LEAF0;
              root_pf(t) <= '0';
              inflight(t) <= '0';
            end loop;
            -- thread 0 walks trees 0, 2, 4, ... and thread 1 trees 1, 3, 5, ...
            active    <= (others => '0');
            n_left(0) <= count - count / 2 + 1;
            n_left(1) <= count / 2 + 1;
            if count > 0 then
              active(0) <= '1';
            end if;
            if count > 1 then
              active(1) <= '1';
            end if;
            ridx(0) <= ridx_add(1, 2);
            ridx(1) <= ridx_add(2, 2);
            state   <= S_ROOT0;

          when S_ROOT0 =>
            next_root(0) <= roots_q;
            state        <= S_ROOT1;

          when S_ROOT1 =>
            next_root(1) <= roots_q;
            ph    <= 0;
            state <= S_RUN;

          when S_RUN =>
            still := '0';
            for t in 0 to 1 loop
              if t = ph then
                -- select: the read of the address selected by the process above is issued in this cycle
                inflight(t) <= '0';
                if active(t) = '1' then
                  if node(t).is_leaf = '1' then
                    acc(t) <= acc(t) + node(t).score;
                    if n_left(t) > 1 then
                      n_left(t)   <= n_left(t) - 1;
                      inflight(t) <= '1';
                      root_pf(t)  <= '1';
                      ridx(t)     <= ridx_add(ridx(t), 2);
                      still       := '1';
                    else
                      n_left(t) <= (others => '0');
                      active(t) <= '0';
                    end if;
                  else
                    inflight(t) <= '1';
                    still       := '1';
                  end if;
                end if;
              else
                -- compare: the node read in the last cycle arrives
                if inflight(t) = '1' then
                  node(t)    <= from_std_logic_vector(nodes_q);
                  go_left(t) <= compare(from_std_logic_vector(nodes_q), X);
                end if;
                if root_pf(t) = '1' then
                  next_root(t) <= roots_q;
                  root_pf(t)   <= '0';
                end if;
                if active(t) = '1' then
                  still := '1';
                end if;
              end if;
            end loop;
            ph <= 1 - ph;
            if still = '0' then
              done_r <= '1';
              state  <= S_IDLE;
            end if;
        end case;
      end if;
    end if;
  end process;

end architecture;
