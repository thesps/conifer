library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

library work;
use work.Constants.all;
use work.Types.all;
--use libBDT.Tree;
--use libBDT.AddReduce;

entity BDT is
  generic(
    iFeature : intArray2DnNodes(0 to nTrees-1);
    iChildLeft : intArray2DnNodes(0 to nTrees-1);
    iChildRight : intArray2DnNodes(0 to nTrees-1);
    iParent : intArray2DnNodes(0 to nTrees-1);
    iLeaf : intArray2DnLeaves(0 to nTrees-1);
    depth : intArray2DnNodes(0 to nTrees-1);
    threshold : ttArray2DnNodes(0 to nTrees-1);
    value : tyArray2DnNodes(0 to nTrees-1);
    initPredict : ty;
    normalisation : ty
  );
  port(
    clk : in std_logic;  -- clock
    X : in txArray(0 to nFeatures-1);           -- input features
    X_vld : in boolean; -- input valid
    y : out ty;           -- output score
    y_vld : out boolean -- output valid
  );
end BDT;

architecture rtl of BDT is
  signal yTrees : tyArray(0 to nTrees-1); -- The score output by each tree
  signal yV : tyArray(0 to 0); -- A vector container
  signal y_vld_arr : boolArray(0 to nTrees-1);
  signal y_vld_arr_v: boolArray(0 to 0); -- A container
  signal ySum : ty; -- The sum of the trees
  signal y_vld_sum : boolean;
  signal yUnnormalised : ty; -- The sum of the trees and the initial predict
  signal y_vld_unnormalised : boolean;
begin

  -- Make all the tree instances
  TreeGen: for i in 0 to nTrees-1 generate
    Treei : entity work.Tree
    generic map(
      iFeature => iFeature(i),
      iChildLeft => iChildLeft(i),
      iChildRight => iChildRight(i),
      iParent => iParent(i),
      iLeaf => iLeaf(i),
      depth => depth(i),
      threshold => threshold(i),
      value => value(i)
    )port map(clk => clk, X => X, X_vld => X_vld, y => yTrees(i), y_vld => y_vld_arr(i));
  end generate;

  -- Sum the output scores using the add tree-reduce
  AddTree : entity work.AddReduce
  generic map(id => "0", depth => 0, pipelining => adder_pipelining)
  port map(clk => clk, d => yTrees, d_vld => y_vld_arr, q => yV, q_vld => y_vld_arr_v);
  ySum <= yV(0);
  y_vld_sum <= y_vld_arr_v(0);

  -- Add the initial predict after the sum of the trees, as the next adder stage of the AddReduce
  GenInitPredict:
  if nStages(nTrees) mod adder_pipelining = 0 generate
    process(clk)
    begin
      if rising_edge(clk) then
        yUnnormalised <= add_ty(ySum, initPredict);
        y_vld_unnormalised <= y_vld_sum;
      end if;
    end process;
  else generate
    yUnnormalised <= add_ty(ySum, initPredict);
    y_vld_unnormalised <= y_vld_sum;
  end generate;

  -- Normalise the score, unless the normalisation is exactly equal to 1
  GenNormalisation:
  if normalise generate
    process(clk)
    begin
      if rising_edge(clk) then
        y <= quantise_ty(y_sfixed(yUnnormalised) * y_sfixed(normalisation));
        y_vld <= y_vld_unnormalised;
      end if;
    end process;
  else generate
    y <= yUnnormalised;
    y_vld <= y_vld_unnormalised;
  end generate;
end rtl;
