#pragma once
#include "params.h"
#include <adf.h>

// One symbol per tile, because the tree range is baked into the function and a
// runtime index cannot choose between them. Every tile has the same signature:
// this kernel has only the per-tile merge, so a tile takes features on a
// stream, emits its own partial on its own stream, and the sum is assumed to
// happen off-array.

#define BDT_DECL_PLIO(S)                                                       \
  void bdt_qs_tile_##S(input_stream<bdtm::feat_t> *,                           \
                       output_stream<bdtm::score_t> *);

#define BDT_DECL_ROLE(S) BDT_DECL_PLIO(S)
#define BDT_DECL_ROLE0(S) BDT_DECL_PLIO(S)

void bdt_qs_obl(input_stream<bdtm::feat_t> *xin,
                output_stream<bdtm::score_t> *sout);

#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#define BDT_LADDER_DECL
#include "tile_roles.h"
#undef BDT_LADDER_DECL
#endif
