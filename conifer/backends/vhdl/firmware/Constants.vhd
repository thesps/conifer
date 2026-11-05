library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;
use ieee.fixed_float_types.all;
use ieee.fixed_pkg.all;

package Constants is

  -- conifer insert constants

  -- construct from the integer representation of the fixed point value
  function to_tx(x : integer) return tx;
  function to_tt(t : integer) return tt;
  function to_ty(y : integer) return ty;

  -- the fixed point values of the input, threshold, and score representations
  function x_sfixed(x : tx) return sfixed;
  function t_sfixed(t : tt) return sfixed;
  function y_sfixed(y : ty) return sfixed;
  -- the score representation of a fixed point value with the score format
  function to_ty(y : sfixed) return ty;

  -- sum of two scores, with the score overflow mode
  function add_ty(a, b : ty) return ty;
  -- quantise to the score type, with the score rounding and overflow modes
  function quantise_ty(y : sfixed) return ty;

end package;

package body Constants is

  function to_tx(x : integer) return tx is
  begin
    return to_signed(x, tx'length);
  end to_tx;

  function to_tt(t : integer) return tt is
  begin
    return to_signed(t, tt'length);
  end to_tt;

  function to_ty(y : integer) return ty is
  begin
    return to_signed(y, ty'length);
  end to_ty;

  function x_sfixed(x : tx) return sfixed is
  begin
    return to_sfixed(std_logic_vector(x), tx_high, tx_low);
  end x_sfixed;

  function t_sfixed(t : tt) return sfixed is
  begin
    return to_sfixed(std_logic_vector(t), tt_high, tt_low);
  end t_sfixed;

  function y_sfixed(y : ty) return sfixed is
  begin
    return to_sfixed(std_logic_vector(y), ty_high, ty_low);
  end y_sfixed;

  function to_ty(y : sfixed) return ty is
  begin
    return signed(to_slv(y));
  end to_ty;

  function add_ty(a, b : ty) return ty is
  begin
    return to_ty(resize(y_sfixed(a) + y_sfixed(b), ty_high, ty_low, ty_overflow_style, fixed_truncate));
  end add_ty;

  function quantise_ty(y : sfixed) return ty is
    -- half of the score type LSB
    constant half : sfixed(ty_low downto ty_low - 1) := "01";
  begin
    if ty_round_half_up then
      return to_ty(resize(y + half, ty_high, ty_low, ty_overflow_style, fixed_truncate));
    else
      return to_ty(resize(y, ty_high, ty_low, ty_overflow_style, ty_round_style));
    end if;
  end quantise_ty;

end package body;
