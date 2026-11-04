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

-- Forest Processing Unit HDL: the FPU configuration, types and functions
-- The configuration constants are written by the conifer FPU writer
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package fpu_pkg is
  -- conifer insert constants

  -- X, and the threshold, of each feature
  subtype tx is signed(TBITS - 1 downto 0);
  -- score
  subtype ty is signed(UBITS - 1 downto 0);
  type tx_array is array(natural range <>) of tx;
  type ty_array is array(natural range <>) of ty;
  subtype features_t is tx_array(0 to NFEATURES - 1);

  -- node addresses: signed, -1 for the children of a leaf
  subtype node_addr_t is signed(ADDRBITS - 1 downto 0);

  -- a decision node
  type node_t is record
    threshold   : tx;
    score       : ty;
    feature     : unsigned(FEATBITS - 1 downto 0); -- leaves have feature -2
    child_left  : node_addr_t;
    child_right : node_addr_t;
    iclass      : unsigned(CLASSBITS - 1 downto 0);
    is_leaf     : std_logic;
  end record;

  -- The node memory word packs the node fields, from the least significant bit:
  --   threshold | score | feature | child_left | child_right | iclass | is_leaf
  -- matching NodeWord in fpu.h
  constant T_LO     : integer := 0;
  constant U_LO     : integer := T_LO + TBITS;
  constant F_LO     : integer := U_LO + UBITS;
  constant L_LO     : integer := F_LO + FEATBITS;
  constant R_LO     : integer := L_LO + ADDRBITS;
  constant C_LO     : integer := R_LO + ADDRBITS;
  constant LEAF_BIT : integer := C_LO + CLASSBITS;
  constant NODEBITS : integer := LEAF_BIT + 1;
  subtype node_word_t is std_logic_vector(NODEBITS - 1 downto 0);

  function to_std_logic_vector(n : node_t) return node_word_t;
  function from_std_logic_vector(w : node_word_t) return node_t;

  -- Tree Engine memory ports: the node memory, and the roots memory holding the number of trees then their root node addresses
  subtype nodes_addr_t is std_logic_vector(NAWIDTH - 1 downto 0);
  subtype roots_addr_t is std_logic_vector(RAWIDTH - 1 downto 0);
  subtype roots_word_t is std_logic_vector(ADDRBITS - 1 downto 0);
  type nodes_addr_array is array(natural range <>) of nodes_addr_t;
  type nodes_word_array is array(natural range <>) of node_word_t;
  type roots_addr_array is array(natural range <>) of roots_addr_t;
  type roots_word_array is array(natural range <>) of roots_word_t;

  -- number of bits to address n words, at least 1
  function clog2(n : integer) return integer;
  function imin(a, b : integer) return integer;
end package;

package body fpu_pkg is
  function to_std_logic_vector(n : node_t) return node_word_t is
    variable w : node_word_t;
  begin
    w(U_LO - 1 downto T_LO)     := std_logic_vector(n.threshold);
    w(F_LO - 1 downto U_LO)     := std_logic_vector(n.score);
    w(L_LO - 1 downto F_LO)     := std_logic_vector(n.feature);
    w(R_LO - 1 downto L_LO)     := std_logic_vector(n.child_left);
    w(C_LO - 1 downto R_LO)     := std_logic_vector(n.child_right);
    w(LEAF_BIT - 1 downto C_LO) := std_logic_vector(n.iclass);
    w(LEAF_BIT)                 := n.is_leaf;
    return w;
  end function;

  function from_std_logic_vector(w : node_word_t) return node_t is
    variable n : node_t;
  begin
    n.threshold   := signed(w(U_LO - 1 downto T_LO));
    n.score       := signed(w(F_LO - 1 downto U_LO));
    n.feature     := unsigned(w(L_LO - 1 downto F_LO));
    n.child_left  := signed(w(R_LO - 1 downto L_LO));
    n.child_right := signed(w(C_LO - 1 downto R_LO));
    n.iclass      := unsigned(w(LEAF_BIT - 1 downto C_LO));
    n.is_leaf     := w(LEAF_BIT);
    return n;
  end function;

  function clog2(n : integer) return integer is
    variable r : integer := 1;
  begin
    while 2**r < n loop
      r := r + 1;
    end loop;
    return r;
  end function;

  function imin(a, b : integer) return integer is
  begin
    if a < b then
      return a;
    end if;
    return b;
  end function;
end package body;
