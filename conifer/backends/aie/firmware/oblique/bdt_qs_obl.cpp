#include "bdt_qs_obl.hpp"
#include "bdt_qs_oblique.hpp"

using namespace adf;
using namespace bdtm;
using namespace bdtv;

template <unsigned SHARD>
__attribute__((always_inline)) static inline vscore
score_shard(input_stream<feat_t> *xin) {
  return qs_score_group<bdtmt::t_begin(SHARD), bdtmt::t_count(SHARD),
                        bdtmt::adds_init(SHARD)>(
      [&](unsigned)
          __attribute__((always_inline)) { return readincr_v<W>(xin); });
}

#define BDT_DEF_PLIO(S)                                                        \
  void bdt_qs_tile_##S(input_stream<feat_t> *__restrict xin,                   \
                       output_stream<score_t> *__restrict sout) {              \
    writeincr(sout, score_shard<S>(xin));                                      \
  }

#define BDT_DEF_ROLE(S) BDT_DEF_PLIO(S)
#define BDT_DEF_ROLE0(S) BDT_DEF_PLIO(S)

void bdt_qs_obl(input_stream<feat_t> *__restrict xin,
                output_stream<score_t> *__restrict sout) {
  writeincr(sout, score_shard<0>(xin));
}

#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#define BDT_LADDER_DEF
#include "tile_roles.h"
#undef BDT_LADDER_DEF
#endif
