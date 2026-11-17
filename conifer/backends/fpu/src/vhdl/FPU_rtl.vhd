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

-- The FPU for RTL designs: the decision function FPU_df with its Tree Engine memories, and a configuration port to
-- load and read back the model, independent of any bus protocol.
--
-- Inference, on clk: FPU_df's X, X_vld, X_rdy, y, y_vld (see FPU_df). The model must not change during inference.
--
-- Configuration, on cfg_clk, which may be asynchronous to clk. One access at a time: cfg_req for one clock starts an
-- access of cfg_addr, writing cfg_wdata if cfg_we, unless cfg_busy. cfg_ack for one clock ends it, with cfg_rdata for a
-- read, and cfg_err if the address is not in the memory map or is read only. The memory map is in fpu_rtl_pkg.
-- The access crosses to clk with a toggle handshake: the cfg2fpu_*_h registers hold the access while it crosses to the
-- cfg2fpu_*_s registers, and fpu2cfg_*_h hold the reply while it crosses to fpu2cfg_*_s. FPU_rtl_cdc.xdc constrains
-- the crossings. An access takes about 3 clocks of each clock domain each way.
-- rst and cfg_rst reset each side's state; an access in flight when either is reset may be lost.
-- READBACK: the node and roots memories can be read through the configuration port. Without, reading them is an
-- error, and the node memories take up to half the block RAMs (see FPU_mem). The INFO words can always be read.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fpu_pkg.all;
use work.fpu_rtl_pkg.all;

entity FPU_rtl is
  generic(
    NODES_RAM_STYLE : string := "block";
    ROOTS_RAM_STYLE : string := "distributed";
    READBACK        : boolean := true
  );
  port(
    clk       : in  std_logic;
    rst       : in  std_logic;
    X         : in  features_t;
    X_vld     : in  std_logic;
    X_rdy     : out std_logic;
    y         : out ty;
    y_vld     : out std_logic;
    cfg_clk   : in  std_logic;
    cfg_rst   : in  std_logic;
    cfg_req   : in  std_logic;
    cfg_we    : in  std_logic;
    cfg_addr  : in  cfg_addr_t;
    cfg_wdata : in  cfg_data_t;
    cfg_busy  : out std_logic;
    cfg_ack   : out std_logic;
    cfg_rdata : out cfg_data_t;
    cfg_err   : out std_logic
  );
end entity;

architecture rtl of FPU_rtl is
  attribute async_reg : string;

  -- configuration clock domain
  signal cfg2fpu_addr_h  : cfg_addr_t := (others => '0');
  signal cfg2fpu_data_h  : cfg_data_t := (others => '0');
  signal cfg2fpu_we_h    : std_logic := '0';
  signal cfg2fpu_req_h   : std_logic := '0';                     -- toggles for each access
  signal fpu2cfg_ack_s   : std_logic_vector(0 to 2) := (others => '0');
  attribute async_reg of fpu2cfg_ack_s : signal is "true";
  signal fpu2cfg_data_s  : cfg_data_t := (others => '0');
  signal fpu2cfg_err_s   : std_logic := '0';
  signal pending         : std_logic := '0';
  signal ack_r           : std_logic := '0';

  -- FPU clock domain
  signal cfg2fpu_req_s   : std_logic_vector(0 to 2) := (others => '0');
  attribute async_reg of cfg2fpu_req_s : signal is "true";
  signal cfg2fpu_addr_s  : cfg_addr_t := (others => '0');
  signal cfg2fpu_data_s  : cfg_data_t := (others => '0');
  signal cfg2fpu_we_s    : std_logic := '0';
  signal fpu2cfg_ack_h   : std_logic := '0';                     -- toggles for each reply
  signal fpu2cfg_data_h  : cfg_data_t := (others => '0');
  signal fpu2cfg_err_h   : std_logic := '0';

  type state_t is (S_IDLE, S_EXEC, S_MEM, S_WAIT, S_SEL, S_REPLY);
  signal state     : state_t := S_IDLE;
  -- the access, decoded
  signal region    : integer range 0 to 3 := 0;
  signal te        : integer range 0 to 2**TE_BITS - 1 := 0;
  signal slot      : unsigned(SLOT_BITS - 1 downto 0) := (others => '0');
  signal word      : integer range 0 to 2**WORD_BITS - 1 := 0;
  signal info_idx  : integer range 0 to 2**REGION_LO - 1 := 0;
  signal valid     : boolean := false;
  -- the node words held until the last word of a node is written
  type stage_t is array(0 to NODE_WORDS - 1) of cfg_data_t;
  signal stage     : stage_t := (others => (others => '0'));
  -- the configuration ports of the memories: one enable per Tree Engine, the rest shared
  signal nb_en, nb_we, rb_en, rb_we : std_logic_vector(0 to NTE - 1) := (others => '0');
  signal nb_addr   : nodes_addr_t := (others => '0');
  signal rb_addr   : roots_addr_t := (others => '0');
  signal nb_d      : node_word_t := (others => '0');
  signal rb_d      : roots_word_t := (others => '0');
  signal nb_q      : nodes_word_array(0 to NTE - 1);
  signal rb_q      : roots_word_array(0 to NTE - 1);
  signal sel_q     : std_logic_vector(NODE_WORDS * CFG_DBITS - 1 downto 0) := (others => '0');

  -- FPU_df's memory ports
  signal na : nodes_addr_array(0 to NTE - 1);
  signal ne : std_logic_vector(0 to NTE - 1);
  signal nq : nodes_word_array(0 to NTE - 1);
  signal ra : roots_addr_array(0 to NTE - 1);
  signal re : std_logic_vector(0 to NTE - 1);
  signal rq : roots_word_array(0 to NTE - 1);
begin

  -------------------------------------------------------------------------------------------------------------------
  -- configuration clock domain: hold the access and toggle the request, then wait for the reply
  cfg_busy  <= pending;
  cfg_ack   <= ack_r;
  cfg_rdata <= fpu2cfg_data_s;
  cfg_err   <= fpu2cfg_err_s;

  process(cfg_clk)
  begin
    if rising_edge(cfg_clk) then
      fpu2cfg_ack_s <= fpu2cfg_ack_h & fpu2cfg_ack_s(0 to 1);
      ack_r <= '0';
      if cfg_rst = '1' then
        pending <= '0';
      elsif pending = '0' then
        if cfg_req = '1' then
          cfg2fpu_addr_h <= cfg_addr;
          cfg2fpu_data_h <= cfg_wdata;
          cfg2fpu_we_h   <= cfg_we;
          cfg2fpu_req_h  <= not cfg2fpu_req_h;
          pending        <= '1';
        end if;
      elsif fpu2cfg_ack_s(1) /= fpu2cfg_ack_s(2) then
        -- the reply is held by fpu2cfg_*_h since before the toggle
        fpu2cfg_data_s <= fpu2cfg_data_h;
        fpu2cfg_err_s  <= fpu2cfg_err_h;
        ack_r          <= '1';
        pending        <= '0';
      end if;
    end if;
  end process;

  -------------------------------------------------------------------------------------------------------------------
  -- FPU clock domain: capture the access, write or read the memories, then hold the reply and toggle the ack
  process(clk)
    variable a : unsigned(CFG_ABITS - 1 downto 0);
    variable v : boolean;
  begin
    if rising_edge(clk) then
      cfg2fpu_req_s <= cfg2fpu_req_h & cfg2fpu_req_s(0 to 1);
      nb_en <= (others => '0');
      nb_we <= (others => '0');
      rb_en <= (others => '0');
      rb_we <= (others => '0');
      if rst = '1' then
        state <= S_IDLE;
      else
        case state is
          when S_IDLE =>
            if cfg2fpu_req_s(1) /= cfg2fpu_req_s(2) then
              -- the access is held by cfg2fpu_*_h since before the toggle
              cfg2fpu_addr_s <= cfg2fpu_addr_h;
              cfg2fpu_data_s <= cfg2fpu_data_h;
              cfg2fpu_we_s   <= cfg2fpu_we_h;
              state          <= S_EXEC;
            end if;

          when S_EXEC =>
            a        := unsigned(cfg2fpu_addr_s);
            region   <= to_integer(a(CFG_ABITS - 1 downto REGION_LO));
            te       <= to_integer(a(REGION_LO - 1 downto TE_LO));
            slot     <= a(TE_LO - 1 downto SLOT_LO);
            word     <= to_integer(a(SLOT_LO - 1 downto WORD_LO));
            info_idx <= to_integer(a(REGION_LO - 1 downto 0));
            state    <= S_MEM;

          when S_MEM =>
            -- check the address, and start the memory access
            v := false;
            case region is
              when REGION_INFO =>
                v := info_idx < N_INFO and cfg2fpu_we_s = '0';
              when REGION_ROOTS =>
                v := te < NTE and slot < 2**RAWIDTH and word = 0 and (READBACK or cfg2fpu_we_s = '1');
                if v then
                  rb_en(te) <= '1';
                  rb_we(te) <= cfg2fpu_we_s;
                end if;
              when REGION_NODES =>
                v := te < NTE and slot < 2**NAWIDTH and word < NODE_WORDS and (READBACK or cfg2fpu_we_s = '1');
                if v then
                  if cfg2fpu_we_s = '1' and word < NODE_WORDS - 1 then
                    stage(word) <= cfg2fpu_data_s;
                  else
                    nb_en(te) <= '1';
                    nb_we(te) <= cfg2fpu_we_s;
                  end if;
                end if;
              when others =>
            end case;
            valid   <= v;
            nb_addr <= std_logic_vector(slot(NAWIDTH - 1 downto 0));
            rb_addr <= std_logic_vector(slot(RAWIDTH - 1 downto 0));
            rb_d    <= cfg2fpu_data_s(ADDRBITS - 1 downto 0);
            for k in 0 to NODE_WORDS - 2 loop
              nb_d((k + 1) * CFG_DBITS - 1 downto k * CFG_DBITS) <= stage(k);
            end loop;
            nb_d(NODEBITS - 1 downto (NODE_WORDS - 1) * CFG_DBITS) <=
              cfg2fpu_data_s(NODEBITS - (NODE_WORDS - 1) * CFG_DBITS - 1 downto 0);
            state   <= S_WAIT;

          when S_WAIT =>
            -- the memories read
            state <= S_SEL;

          when S_SEL =>
            -- select the Tree Engine
            sel_q <= (others => '0');
            if valid and region = REGION_NODES then
              sel_q(NODEBITS - 1 downto 0) <= nb_q(te);
            elsif valid and region = REGION_ROOTS then
              sel_q(ADDRBITS - 1 downto 0) <= rb_q(te);
            end if;
            state <= S_REPLY;

          when S_REPLY =>
            fpu2cfg_data_h <= (others => '0');
            fpu2cfg_err_h  <= '1';
            if valid then
              fpu2cfg_err_h <= '0';
              if region = REGION_INFO then
                fpu2cfg_data_h <= info_word(info_idx);
              else
                fpu2cfg_data_h <= sel_q((word + 1) * CFG_DBITS - 1 downto word * CFG_DBITS);
              end if;
            end if;
            fpu2cfg_ack_h <= not fpu2cfg_ack_h;
            state         <= S_IDLE;
        end case;
      end if;
    end if;
  end process;

  -------------------------------------------------------------------------------------------------------------------
  Mems : for i in 0 to NTE - 1 generate
    Nodes : entity work.FPU_mem
      generic map(WIDTH => NODEBITS, ABITS => NAWIDTH, STYLE => NODES_RAM_STYLE, B_READ => READBACK)
      port map(clk => clk, a_en => ne(i), a_addr => na(i), a_q => nq(i),
               b_en => nb_en(i), b_we => nb_we(i), b_addr => nb_addr, b_d => nb_d, b_q => nb_q(i));
    Roots : entity work.FPU_mem
      generic map(WIDTH => ADDRBITS, ABITS => RAWIDTH, STYLE => ROOTS_RAM_STYLE, B_READ => READBACK)
      port map(clk => clk, a_en => re(i), a_addr => ra(i), a_q => rq(i),
               b_en => rb_en(i), b_we => rb_we(i), b_addr => rb_addr, b_d => rb_d, b_q => rb_q(i));
  end generate;

  DF : entity work.FPU_df
    port map(
      clk => clk, rst => rst, X => X, X_vld => X_vld, X_rdy => X_rdy, y => y, y_vld => y_vld,
      nodes_addr => na, nodes_en => ne, nodes_q => nq,
      roots_addr => ra, roots_en => re, roots_q => rq
    );

end architecture;
