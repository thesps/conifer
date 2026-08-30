#pragma once
#include "params.h"
#include <adf.h>

#define BDT_DECL_PLIO(S)                                                       \
  void bdt_qs_tile_##S(input_stream<bdtm::feat_t> *,                           \
                       output_stream<bdtm::score_t> *);
#define BDT_DECL_BUF(S)                                                        \
  void bdt_qs_tile_##S(adf::input_buffer<bdtm::feat_t> &,                      \
                       output_stream<bdtm::score_t> *);

#if BDT_FEED_MEMTILE
#define BDT_DECL_ROLE(S) BDT_DECL_BUF(S)
#else
#define BDT_DECL_ROLE(S) BDT_DECL_PLIO(S)
#endif
#define BDT_DECL_ROLE0(S) BDT_DECL_ROLE(S)

void bdt_qs_mt(input_stream<bdtm::feat_t> *xin,
               output_stream<bdtm::score_t> *sout);

// generated per project by roles.py
#define BDT_LADDER_DECL
#include "tile_roles.h"
#undef BDT_LADDER_DECL
