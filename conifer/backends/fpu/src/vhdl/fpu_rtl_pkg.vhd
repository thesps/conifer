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

-- The memory map of FPU_rtl's configuration port: 32 bit words, word addressed. The address fields, from the least
-- significant bit:
--   word | slot | Tree Engine | region
-- region INFO  (0): read only words describing the FPU, at slot 0, Tree Engine 0, word = the index in INFO_*
-- region ROOTS (1): slot r of Tree Engine t's roots memory: the number of trees, then the root node addresses. word 0
-- region NODES (2): word k of node slot of Tree Engine t's node memory. A node is NODE_WORDS words, least significant
--                   first: words 0 to NODE_WORDS - 2 are held, and the write of word NODE_WORDS - 1 writes the node
-- Reads return the memory contents, zero extended. The Python mirror is conifer.backends.fpu.memory_map.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fpu_pkg.all;

package fpu_rtl_pkg is
  constant CFG_DBITS     : integer := 32;
  constant NODE_WORDS    : integer := (NODEBITS + CFG_DBITS - 1) / CFG_DBITS;
  constant WORD_BITS     : integer := clog2(NODE_WORDS);
  -- the larger of the node and roots memory address widths
  constant SLOT_BITS     : integer := NAWIDTH + RAWIDTH - imin(NAWIDTH, RAWIDTH);
  constant TE_BITS       : integer := clog2(NTE);
  constant WORD_LO       : integer := 0;
  constant SLOT_LO       : integer := WORD_LO + WORD_BITS;
  constant TE_LO         : integer := SLOT_LO + SLOT_BITS;
  constant REGION_LO     : integer := TE_LO + TE_BITS;
  constant CFG_ABITS     : integer := REGION_LO + 2;
  constant REGION_INFO   : integer := 0;
  constant REGION_ROOTS  : integer := 1;
  constant REGION_NODES  : integer := 2;

  -- the INFO words
  constant INFO_VERSION  : integer := 0;  -- of this memory map
  constant INFO_NTE      : integer := 1;
  constant INFO_NNODES   : integer := 2;
  constant INFO_NROOTS   : integer := 3;
  constant INFO_NFEAT    : integer := 4;
  constant INFO_TBITS    : integer := 5;
  constant INFO_UBITS    : integer := 6;
  constant INFO_NODEBITS : integer := 7;
  constant N_INFO        : integer := 8;
  constant MAP_VERSION   : integer := 1;

  subtype cfg_addr_t is std_logic_vector(CFG_ABITS - 1 downto 0);
  subtype cfg_data_t is std_logic_vector(CFG_DBITS - 1 downto 0);

  function info_word(i : integer) return cfg_data_t;
end package;

package body fpu_rtl_pkg is
  function info_word(i : integer) return cfg_data_t is
    variable v : integer := 0;
  begin
    case i is
      when INFO_VERSION  => v := MAP_VERSION;
      when INFO_NTE      => v := NTE;
      when INFO_NNODES   => v := NNODES;
      when INFO_NROOTS   => v := NROOTS;
      when INFO_NFEAT    => v := NFEATURES;
      when INFO_TBITS    => v := TBITS;
      when INFO_UBITS    => v := UBITS;
      when INFO_NODEBITS => v := NODEBITS;
      when others        => v := 0;
    end case;
    return std_logic_vector(to_unsigned(v, CFG_DBITS));
  end function;
end package body;
