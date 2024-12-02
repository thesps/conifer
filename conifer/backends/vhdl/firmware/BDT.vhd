library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

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
    threshold : txArray2DnNodes(0 to nTrees-1);
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
  signal yTrees : tyArray(0 to nTrees); -- The score output by each tree (1 extra element for the initial predict)
  signal yV : tyArray(0 to 0); -- A vector container
  signal y_vld_arr : boolArray(0 to nTrees);
  signal y_vld_arr_v: boolArray(0 to 0); -- A container
  signal yNormalisedWide : signed(ty'length * 2 -1 downto 0);
  signal yUnnormalised : ty;
  signal y_vld_unnormalised : boolean;
begin

  yTrees(nTrees) <= initPredict;
  y_vld_arr(nTrees) <= true;

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
  port map(clk => clk, d => yTrees, d_vld => y_vld_arr, q => yV, q_vld => y_vld_arr_v);
  yUnnormalised <= yV(0);
  y_vld_unnormalised <= y_vld_arr_v(0);

  -- Normalise the score, unless the normalisation is exactly equal to '1'
  GenNormalisation:
  if (normalisation = 1) generate
    y <= yUnnormalised;
    y_vld <= y_vld_unnormalised;
  else generate
    process(clk)
      begin
      if rising_edge(clk) then
        yNormalisedWide <= yUnnormalised * normalisation;
        y_vld <= y_vld_unnormalised;
      end if;
    end process;
    y <= yNormalisedWide(norm_slice_h downto norm_slice_l);
  end generate;
end rtl;
