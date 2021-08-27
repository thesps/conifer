library ieee;
  use ieee.std_logic_1164.all;
  use ieee.std_logic_misc.all;
  use ieee.numeric_std.all;

  use work.Constants.all;
  use work.Types.all;
  package Arrays0 is

    constant initPredict : ty := to_ty(-2491);
    constant feature : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((1, 1, 1, -2, -2, 5, -2, -2, 2, 1, -2, -2, -2, -2, -2),
                (5, 3, 9, -2, -2, 9, -2, -2, 7, 2, -2, -2, 3, -2, -2),
                (2, 2, 3, -2, -2, 8, -2, -2, 7, 8, -2, -2, 4, -2, -2),
                (3, 6, 0, -2, -2, 7, -2, -2, 9, 2, -2, -2, 0, -2, -2),
                (0, 9, 8, -2, -2, 6, -2, -2, 0, 9, -2, -2, 7, -2, -2),
                (8, 6, 6, -2, -2, 4, -2, -2, 5, 8, -2, -2, 3, -2, -2),
                (3, 3, 8, -2, -2, 9, -2, -2, 9, 9, -2, -2, 9, -2, -2),
                (5, 3, 4, -2, -2, 6, -2, -2, 9, 2, -2, -2, 8, -2, -2),
                (7, 7, 1, -2, -2, 5, -2, -2, 5, 4, -2, -2, 5, -2, -2),
                (2, 2, 6, -2, -2, 8, -2, -2, 2, 6, -2, -2, 2, -2, -2),
                (6, 6, 4, -2, -2, 1, -2, -2, 4, 6, -2, -2, 6, -2, -2),
                (1, 0, 1, -2, -2, 6, -2, -2, 0, 0, -2, -2, 4, -2, -2),
                (8, 8, 5, -2, -2, 3, -2, -2, 7, 8, -2, -2, 6, -2, -2),
                (5, 0, 4, -2, -2, 3, -2, -2, 5, 4, -2, -2, 9, -2, -2),
                (2, 2, 4, -2, -2, 7, -2, -2, 1, 2, -2, -2, 4, -2, -2),
                (0, 7, 0, -2, -2, 9, -2, -2, 0, -2, 4, -2, -2, -2, -2),
                (7, 5, 3, -2, -2, 1, -2, -2, 3, 9, -2, -2, 4, -2, -2),
                (9, 1, 1, -2, -2, 7, -2, -2, 1, 1, -2, -2, 9, -2, -2),
                (4, 2, 2, -2, -2, 9, -2, -2, 8, 3, -2, -2, 9, -2, -2),
                (2, 2, 9, -2, -2, 7, -2, -2, 6, 8, -2, -2, 4, -2, -2)
                );
    constant threshold_int : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((73288, -89909, -138221, -131072, -131072, 81380, -131072, -131072, 61786, 100196, -131072, -131072, -131072, -131072, -131072),
                (-89622, -58505, -14062, -131072, -131072, 51557, -131072, -131072, 80643, 109513, -131072, -131072, -53561, -131072, -131072),
                (-78882, -117919, 57547, -131072, -131072, 88475, -131072, -131072, -107815, -37326, -131072, -131072, -106379, -131072, -131072),
                (110561, 116424, -105196, -131072, -131072, 38426, -131072, -131072, 15994, 58226, -131072, -131072, 26521, -131072, -131072),
                (100961, 129081, -95887, -131072, -131072, -54758, -131072, -131072, 139542, 63818, -131072, -131072, 89875, -131072, -131072),
                (77182, -94226, -124305, -131072, -131072, 76979, -131072, -131072, 38262, 129134, -131072, -131072, -8692, -131072, -131072),
                (-81386, -130857, 89189, -131072, -131072, -65361, -131072, -131072, -79519, -151384, -131072, -131072, 77317, -131072, -131072),
                (116180, 83532, -65682, -131072, -131072, 63769, -131072, -131072, -4299, 84156, -131072, -131072, 31536, -131072, -131072),
                (117079, -86930, 30552, -131072, -131072, -135855, -131072, -131072, -46773, 47192, -131072, -131072, 2591, -131072, -131072),
                (64468, -55150, -53468, -131072, -131072, -145071, -131072, -131072, 193053, -64288, -131072, -131072, 216245, -131072, -131072),
                (54682, -53483, 57210, -131072, -131072, 71172, -131072, -131072, 113710, 97929, -131072, -131072, 93514, -131072, -131072),
                (-65504, 88362, -119082, -131072, -131072, 63691, -131072, -131072, -81752, -146908, -131072, -131072, -143962, -131072, -131072),
                (66417, -63652, 57579, -131072, -131072, -86240, -131072, -131072, -23690, 165442, -131072, -131072, -184247, -131072, -131072),
                (-71954, 15730, 82989, -131072, -131072, 57112, -131072, -131072, 61217, 132908, -131072, -131072, 33446, -131072, -131072),
                (-135976, -166947, 53805, -131072, -131072, -41649, -131072, -131072, 135470, 98534, -131072, -131072, 61586, -131072, -131072),
                (97880, 96085, -44992, -131072, -131072, 38875, -131072, -131072, 98452, -131072, -49601, -131072, -131072, -131072, -131072),
                (-169574, -36430, 2791, -131072, -131072, -62878, -131072, -131072, 61935, -106500, -131072, -131072, -77552, -131072, -131072),
                (157886, 58585, -44393, -131072, -131072, -33979, -131072, -131072, 22876, -55585, -131072, -131072, 162198, -131072, -131072),
                (-122824, -73769, -83994, -131072, -131072, -33521, -131072, -131072, 104671, -142185, -131072, -131072, 60271, -131072, -131072),
                (-107388, -107920, 112319, -131072, -131072, -21719, -131072, -131072, -119257, -110048, -131072, -131072, 58940, -131072, -131072)
                );
    constant value_int : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((0, -2523, 18297, 127367, 54726, -4733, -27695, 62407, 15432, 13043, 22979, 83348, 133611, 133611, 133611),
                (432, 19958, 33508, 169412, 124945, 16272, 49324, 143318, -1254, -3225, -20482, 113763, 15569, 133236, 48438),
                (369, 15067, 26099, 109015, 182911, 10301, 36572, 157064, -1654, 21447, 158816, 83869, -2959, 96495, -20814),
                (244, -776, -1773, 100107, -15808, 24046, 141203, 84130, 23520, 18709, 106492, 8453, 30954, 126051, 221835),
                (286, -950, -1700, 80032, -17066, 28832, 241440, 144067, 18352, 12949, 50362, 170509, 35260, 194996, 84168),
                (169, -1694, 14901, 147525, 62871, -2810, -25413, 61844, 12979, 8967, 30239, 140688, 23321, 164936, 95762),
                (111, 11616, 26331, 160618, 80989, 8432, 155376, 38484, -1225, 10297, 154058, 51333, -2503, -24083, 46876),
                (139, -573, -1416, 32076, -17987, 7771, 39071, 154434, 16782, 8640, 48035, 214302, 25332, 141749, 97332),
                (157, -281, 7680, 30275, 108086, -1126, 112586, -10375, 14068, 29908, 152634, 84192, 10583, 22218, 108218),
                (204, -766, 4349, 88456, 19620, -2309, 146606, -19280, 5902, 5114, 97529, 30004, 30598, 171030, 108693),
                (98, -1038, 3123, 13316, 91609, -2420, -25329, 28842, 4548, 3940, 15813, 64573, 18495, 181373, 17936),
                (28, 6014, 5167, 83469, 28139, 13810, 120314, 24621, -1012, 4750, 119576, 26371, -1765, 110442, -17189),
                (170, -624, 3382, 16722, 80328, -1703, 47212, -21639, 4292, 8857, 63299, 229867, 2304, 302346, 18601),
                (49, 4707, 1311, 1808, 130712, 10289, 74675, 114828, -668, -1518, -16352, 123678, 2986, 11329, 60223),
                (106, 14020, 31114, 175620, 85294, 10357, 123213, 60293, -135, -353, -7603, 60871, 12223, 115898, 52613),
                (128, -299, -726, 18305, -18417, 5622, 36495, 109786, 5849, 205906, 5568, 106867, 44529, 205906, 205906),
                (71, 21430, 40913, 218325, 127230, 13080, 120663, 85441, -36, -658, 63307, -10859, 3360, 115551, 24085),
                (69, -55, -716, 17391, -17847, 2717, 73795, 12748, 15378, 23921, 68968, 153488, 6835, 115290, 73743),
                (51, 7181, 49117, 704458, 160616, 5568, -22358, 78594, -147, -438, 138398, -6611, 4747, 35653, 117968),
                (103, 4937, 4386, 43016, 101485, 35842, 191698, 116132, -190, 6819, 197183, 71388, -377, -10403, 23506)
                );
    constant children_left : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 13, 11, -1, -1, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1),
                (1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1)
                );
    constant children_right : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 10, 14, 12, -1, -1, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1),
                (8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1)
                );
    constant parent : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 8, 10, 10, 9, 9),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12),
                (-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12)
                );
    constant depth : intArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := ((0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 2, 3, 3, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3),
                (0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3)
                );
    constant iLeaf : intArray2D(nTrees-1 downto 0)(nLeaves-1 downto 0) := ((3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 11, 12, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14),
                (3, 4, 6, 7, 10, 11, 13, 14)
                );
    constant value : tyArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_tyArray2D(value_int);
      constant threshold : txArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_txArray2D(threshold_int);
end Arrays0;