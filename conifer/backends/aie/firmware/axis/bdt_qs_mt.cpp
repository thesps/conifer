#include "bdt_qs_mt.hpp"
#include "bdt_qs_pertree_v2.hpp"

using namespace adf;
using namespace bdtm;
using namespace bdtv;

template <unsigned SHARD>
__attribute__((always_inline)) static inline vscore
score_shard(input_stream<feat_t> *xin) {
  return qs_score_group<bdtmt::t_begin(SHARD), bdtmt::t_count(SHARD),
                        bdtmt::adds_init(SHARD), bdtmt::n_feat(SHARD)>(
      [&](unsigned)
          __attribute__((always_inline)) { return readincr_v<W>(xin); });
}

template <unsigned SHARD>
__attribute__((always_inline)) static inline vscore
score_shard_buf(const feat_t *__restrict p) {
  return qs_score_group<bdtmt::t_begin(SHARD), bdtmt::t_count(SHARD),
                        bdtmt::adds_init(SHARD), bdtmt::n_feat(SHARD)>(
      [&](unsigned f)
          __attribute__((always_inline)) { return aie::load_v<W>(p + f * W); });
}

#define BDT_DEF_BUF(S)                                                         \
  void bdt_qs_tile_##S(adf::input_buffer<feat_t> &xin,                         \
                       output_stream<score_t> *__restrict sout) {              \
    writeincr(sout, score_shard_buf<S>(xin.data()));                           \
  }

void bdt_qs_mt(input_stream<feat_t> *__restrict xin,
               output_stream<score_t> *__restrict sout) {
  writeincr(sout, score_shard<0>(xin));
}

#define BDT_DEF_PLIO(S)                                                        \
  void bdt_qs_tile_##S(input_stream<feat_t> *__restrict xin,                   \
                       output_stream<score_t> *__restrict sout) {              \
    writeincr(sout, score_shard<S>(xin));                                      \
  }

#if BDT_FEED_MEMTILE
#define BDT_DEF_ROLE(S) BDT_DEF_BUF(S)
#else
#define BDT_DEF_ROLE(S) BDT_DEF_PLIO(S)
#endif
#define BDT_DEF_ROLE0(S) BDT_DEF_ROLE(S)

#define BDT_LADDER_DEF
#include "tile_roles.h"
#undef BDT_LADDER_DEF
