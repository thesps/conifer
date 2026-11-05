library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;

package Types is

  type intArray is array(natural range <>) of integer;
  subtype intArraynLeaves is intArray(0 to nLeaves - 1); -- nLeaves defined in Constants
  subtype intArraynNodes is intArray(0 to nNodes - 1); -- nLeaves defined in Constants
  type boolArray is array(natural range <>) of boolean;
  subtype boolArraynNodes is boolArray(0 to nNodes - 1); -- nNodes defined in Constants
  type txArray is array(natural range <>) of tx; -- tx defined in Constants
  type ttArray is array(natural range <>) of tt; -- tt defined in Constants
  subtype ttArraynNodes is ttArray(0 to nNodes - 1); -- nNodes defined in Constants
  type tyArray is array(natural range <>) of ty; -- ty defined in Constants
  subtype tyArraynNodes is tyArray(0 to nNodes - 1); -- nNodes defined in Constants

  type intArray2DnNodes is array(natural range <>) of intArraynNodes;
  type intArray2DnLeaves is array(natural range <>) of intArraynLeaves;
  type boolArray2DnNodes is array(natural range <>) of boolArraynNodes;
  type ttArray2DnNodes is array(natural range <>) of ttArraynNodes;
  type tyArray2DnNodes is array(natural range <>) of tyArraynNodes;

  function to_tyArray(yArray : intArray) return tyArray;
  function to_tyArray2D(yArray2D : intArray2DnNodes) return tyArray2DnNodes;
  function to_ttArray(tArray : intArray) return ttArray;
  function to_ttArray2D(tArray2D : intArray2DnNodes) return ttArray2DnNodes;
  -- number of adder stages to reduce n inputs to one
  function nStages(n : integer) return integer;

end Types;

package body Types is

  function to_tyArray(yArray : intArray) return tyArray is
    variable yArrayCast : tyArray(yArray'low to yArray'high);
  begin
    for i in yArray'low to yArray'high loop
        yArrayCast(i) := to_ty(yArray(i));
    end loop;
    return yArrayCast;
  end to_tyArray;

  function to_tyArray2D(yArray2D : intArray2DnNodes) return tyArray2DnNodes is
    variable yArray2DCast : tyArray2DnNodes(yArray2D'low to yArray2D'high);
  begin
    for i in yArray2D'low to yArray2D'high loop
        yArray2DCast(i) := to_tyArray(yArray2D(i));
    end loop;
    return yArray2DCast;
  end to_tyArray2D;

  function to_ttArray(tArray : intArray) return ttArray is
    variable tArrayCast : ttArray(tArray'low to tArray'high);
  begin
    for i in tArray'low to tArray'high loop
        tArrayCast(i) := to_tt(tArray(i));
    end loop;
    return tArrayCast;
  end to_ttArray;

  function to_ttArray2D(tArray2D : intArray2DnNodes) return ttArray2DnNodes is
    variable tArray2DCast : ttArray2DnNodes(tArray2D'low to tArray2D'high);
  begin
    for i in tArray2D'low to tArray2D'high loop
        tArray2DCast(i) := to_ttArray(tArray2D(i));
    end loop;
    return tArray2DCast;
  end to_ttArray2D;

  function nStages(n : integer) return integer is
    variable stages : integer := 0;
    variable remaining : integer := n;
  begin
    while remaining > 1 loop
      remaining := (remaining + 1) / 2;
      stages := stages + 1;
    end loop;
    return stages;
  end nStages;

end Types;
