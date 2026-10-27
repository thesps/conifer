library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

entity Tree is
  generic(
    iFeature : intArray(0 to nNodes-1);
    iChildLeft : intArray(0 to nNodes-1);
    iChildRight : intArray(0 to nNodes-1);
    iParent : intArray(0 to nNodes-1);
    iLeaf : intArray(0 to nLeaves-1);
    depth : intArray(0 to nNodes-1);
    threshold : txArray(0 to nNodes-1);
    value : tyArray(0 to nNodes-1)
  );
  port(
    clk   : in std_logic;  -- clock
    X     : in txArray(0 to nFeatures-1) := (others => to_tx(0));           -- input features
    X_vld : in boolean := false; -- input valid
    y     : out ty := to_ty(0);           -- output score
    y_vld : out boolean := false -- output valid
  );
end tree;

architecture rtl of tree is

  signal comparison : boolArray(0 to nNodes-1) := (others => false);
  signal comparisonPipe : boolArray2DnNodes(0 to maxdepth) := (others => (others => false));
  signal activation : boolArray(0 to nNodes-1) := (others => false);
  signal activation_reg : boolArray(0 to nNodes-1) := (others => false);
  signal vld_pipe : boolArray(0 to 3) := (others => false);

  component Split is
  port(
    a : in tx;
    b : in tx;
    q : out boolean
  );
  end component;

begin

  -- propagate the valid signal
  vld_pipe(0) <= X_vld;
  vld_pipe(1 to 3) <= vld_pipe(0 to 2) when rising_edge(clk);
  y_vld <= vld_pipe(3);

  -- do all the comparisons
  GenComp:
  for i in 0 to nNodes-1 generate
    NonLeaf: if iFeature(i) /= -2 generate
        signal comparison_result : boolean;
      begin
        -- Compare feature for this node to threshold for this node
        -- Split module is defined in PkgSplit
        TheSplit : entity work.Split
        port map(X(iFeature(i)), threshold(i), comparison_result);
      process(clk)
      begin
        if rising_edge(clk) then
          comparison(i) <= comparison_result;
        end if;
      end process;
    end generate NonLeaf;
    -- Leaf nodes don't do comparisons
    Leaf: if iFeature(i) = -2 generate
      process(clk)
      begin
        -- Leaves are always active
        comparison(i) <= true;
      end process;
    end generate Leaf;
  end generate GenComp;

  -- do all the node activations
  -- the root node is always active
  activation(0) <= true; 
  GenAct:
  for i in 1 to nNodes-1 generate
    -- the root node is always active
    LeftChild:
    if i = iChildLeft(iParent(i)) generate
    begin
      activation(i) <= comparison(iParent(i)) and activation(iParent(i));
    end generate LeftChild;
    RightChild:
    if i = iChildRight(iParent(i)) generate
    begin
      activation(i) <= (not comparison(iParent(i))) and activation(iParent(i));  
    end generate RightChild;
  end generate GenAct;

  process(clk)
  begin
    if rising_edge(clk) then
      activation_reg <= activation;
    end if;
  end process;

  -- Assign the score from the active leaf
  GenScore:
  process(clk)
  begin
    if rising_edge(clk) then
     for i in 0 to nLeaves-1 loop
        if activation_reg(iLeaf(i)) then
          y <= value(iLeaf(i));
          exit;
        end if;
        y <= to_ty(0);
      end loop;
    end if;
  end process;

end rtl;
