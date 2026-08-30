#pragma once
#include "bdt_vec.hpp"
#include "parameters.h"
#include <array>
#include <type_traits>

#ifndef BDT_V2_UNROLL
#define BDT_V2_UNROLL 1
#endif

namespace bdtv {

using bvw_t = std::conditional_t<(bdtm::MAX_LEAVES <= 16), uint16_t, uint32_t>;
constexpr unsigned BV_BITS = 8 * sizeof(bvw_t);
constexpr unsigned BV_WORDS = (bdtm::MAX_LEAVES + BV_BITS - 1) / BV_BITS;

using vbvw = aie::vector<bvw_t, W>;

constexpr bvw_t bv_word(uint64_t v, unsigned q) {
  return (bvw_t)((v >> (BV_BITS * q)) & (bvw_t) ~(bvw_t)0);
}

using leaf_t = elem_of<decltype(bdtm::LEAVES)>;
using vleaf = aie::vector<leaf_t, W>;

constexpr unsigned log2_of(unsigned v) {
  unsigned n = 0;
  while ((1u << n) < v)
    n++;
  return n;
}

constexpr unsigned MIN_LEAF_SLOTS = 128u / (8u * sizeof(bdtm::LEAVES[0]));
constexpr unsigned LEAF_SLOTS = [] {
  unsigned n = 1;
  while (n < bdtm::MAX_LEAVES)
    n <<= 1;
  return n < MIN_LEAF_SLOTS ? MIN_LEAF_SLOTS : n;
}();
constexpr unsigned log2_leaves() {
  unsigned n = 0;
  while ((1u << n) < LEAF_SLOTS)
    n++;
  return n;
}
static_assert((1u << log2_leaves()) == LEAF_SLOTS);

constexpr unsigned WORD_BITS_LOG2 = (BV_BITS == 16) ? 4u : 5u;
constexpr unsigned IN_WORD_BITS =
    log2_leaves() < WORD_BITS_LOG2 ? log2_leaves() : WORD_BITS_LOG2;

constexpr bvw_t index_bit_mask(unsigned p) {
  bvw_t mask = 0;
  for (unsigned q = 0; q < BV_BITS; q++)
    if ((q >> p) & 1u)
      mask |= (bvw_t)((bvw_t)1 << q);
  return mask;
}

inline constexpr std::array<bvw_t, IN_WORD_BITS> IDX_MASK = [] {
  std::array<bvw_t, IN_WORD_BITS> a{};
  for (unsigned p = 0; p < IN_WORD_BITS; p++)
    a[p] = index_bit_mask(p);
  return a;
}();

using vrow = aie::vector<leaf_t, LEAF_SLOTS>;

alignas(sizeof(leaf_t) * LEAF_SLOTS) inline constexpr std::array<
    leaf_t, bdtm::N_TREES * LEAF_SLOTS> LEAVES_P = [] {
  std::array<leaf_t, bdtm::N_TREES * LEAF_SLOTS> a{};
  for (unsigned h = 0; h < bdtm::N_TREES; h++)
    for (unsigned j = 0; j < bdtm::MAX_LEAVES; j++)
      a[h * LEAF_SLOTS + j] = bdtm::LEAVES[h * bdtm::MAX_LEAVES + j];
  return a;
}();

template <unsigned Lo, unsigned N>
__attribute__((always_inline)) inline vleaf mux(const vrow &row,
                                                const vmask *mbit) {
  static_assert(N >= 2 && (N & (N - 1)) == 0);
  constexpr unsigned BIT = log2_of(N) - 1;
  if constexpr (N == 2) {
    // The bottom level takes its two candidates as elements of a vector already
    // in a register, so the pair collapses to one vector, avoiding scalar load
    return aie::select(row[Lo], row[Lo + 1], mbit[BIT]);
  } else {
    const vleaf lo = mux<Lo, N / 2>(row, mbit);
    const vleaf hi = mux<Lo + N / 2, N / 2>(row, mbit);
    return aie::select(lo, hi, mbit[BIT]);
  }
}

__attribute__((always_inline)) inline vleaf leaf_value(const vbvw v[BV_WORDS],
                                                       unsigned h) {
  const vbvw zero = aie::zeros<bvw_t, W>();

  // The isolated lowest set bit of the lowest non-zero word, and that word's
  // index. Folded from the top down so a lower non-zero word overrides the
  // words above it.
  vbvw lsb = aie::bit_and(v[BV_WORDS - 1], aie::sub(zero, v[BV_WORDS - 1]));
  vmask wbit[BV_WORDS > 1 ? log2_leaves() - IN_WORD_BITS : 1];
#pragma unroll
  for (int q = (int)BV_WORDS - 2; q >= 0; q--) {
    const vbvw lsbq = aie::bit_and(v[q], aie::sub(zero, v[q]));
    const vmask lower_wins = aie::neq(v[q], (bvw_t)0);
    lsb = aie::select(lsb, lsbq, lower_wins);
  }
  // Which word won, as index bits above the in-word ones. With BV_WORDS at 2,
  // this is a single bit that is set iff word 0 was empty.
#pragma unroll
  for (unsigned p = 0; p + IN_WORD_BITS < log2_leaves(); p++) {
    vmask m = aie::eq(v[0], (bvw_t)0); // only the two-word case is reachable
    wbit[p] = m;
  }

  const vrow row = aie::load_v<LEAF_SLOTS>(&LEAVES_P[h * LEAF_SLOTS]);

  vmask mbit[log2_leaves()];
#pragma unroll
  for (unsigned p = 0; p < IN_WORD_BITS; p++)
    mbit[p] = aie::neq(aie::bit_and(IDX_MASK[p], lsb), (bvw_t)0);
#pragma unroll
  for (unsigned p = IN_WORD_BITS; p < log2_leaves(); p++)
    mbit[p] = wbit[p - IN_WORD_BITS];

  return mux<0, LEAF_SLOTS>(row, mbit);
}

constexpr unsigned QT_N = bdtm::N_TREES * bdtm::QS_NODES_PER_TREE;

constexpr unsigned MIN_QT_LANES = 128u / (8u * sizeof(int16_t));
constexpr unsigned qt_lanes() {
  unsigned n = 1;
  while (n < bdtm::QS_NODES_PER_TREE)
    n <<= 1;
  return n < MIN_QT_LANES ? MIN_QT_LANES : n;
}
constexpr unsigned QT_LANES =
    qt_lanes(); // depth 4: 16, one 256-bit vector of int16

constexpr unsigned QT_HALVES = (QT_LANES * BV_BITS > 1024u) ? 2u : 1u;
constexpr unsigned QT_CHUNK = QT_LANES / QT_HALVES;
// The basis kernel has no chunked node loop. conifer's writer should refuse
// the shapes that would need one before emitting a project
static_assert(QT_HALVES == 1, "the basis kernel has no chunked node loop");

static_assert(QT_CHUNK * BV_BITS <= 1024,
              "one chunk of a tree's bitvector run still exceeds a single "
              "aie::vector");

inline constexpr std::array<int16_t, QT_N + QT_LANES> QT_THR_P = [] {
  std::array<int16_t, QT_N + QT_LANES> a{};
  for (unsigned i = 0; i < QT_N; i++)
    a[i] = bdtm::QT_THR[i];
  return a;
}();

constexpr unsigned QT_STRIDE = QT_N + QT_LANES;
inline constexpr std::array<bvw_t, BV_WORDS * QT_STRIDE> QT_BV_P = [] {
  std::array<bvw_t, BV_WORDS * QT_STRIDE> a{};
  for (unsigned q = 0; q < BV_WORDS; q++) {
    for (unsigned i = 0; i < QT_N; i++)
      a[q * QT_STRIDE + i] = bv_word(bdtm::QT_BV[i], q);
    for (unsigned i = QT_N; i < QT_STRIDE; i++)
      a[q * QT_STRIDE + i] = ~(bvw_t)0;
  }
  return a;
}();

static_assert((bdtm::N_FEATURES & (bdtm::N_FEATURES - 1)) == 0,
              "the weight row is loaded as one aie::vector, which needs a "
              "power-of-two lane count, N_FEATURES is not one");

inline void build_basis(const vfeat *x, vfeat *basis) {
  for (unsigned b = 0; b < bdtm::BASIS_N; b++) {
    vacc a;
    a.from_vector(aie::zeros<feat_t, W>());
    a = aie::mac(a, x[bdtm::BASIS_I[b]], bdtm::BASIS_WI[b]);
    a = aie::mac(a, x[bdtm::BASIS_J[b]], bdtm::BASIS_WJ[b]);
    basis[b] = a.template to_vector<feat_t>(bdtm::WGT_SHIFT);
  }
}

constexpr unsigned TERM_LANES = 64;
static_assert(bdtm::QS_NODES_PER_TREE * bdtm::MAX_TERMS <= TERM_LANES,
              "one tree's term block no longer fits a single vector load");

// The tree range is a template parameter so a tile scores only its own shard
// The basis itself is not subsetted, it's built over the whole feature set on
// every tile
template <unsigned T_BEGIN, unsigned T_COUNT, bool ADD_INIT, typename LoadRow>
__attribute__((always_inline)) inline vscore qs_score_group(LoadRow load_row) {
  static_assert(T_BEGIN + T_COUNT <= bdtm::N_TREES,
                "tree range runs off the ensemble");

  vfeat x[bdtm::N_FEATURES];
  for (unsigned k = 0; k < bdtm::N_FEATURES; k++)
    x[k] = load_row(k);

  vfeat basis[bdtm::BASIS_N];
  build_basis(x, basis);

  const vbvw keep_all = aie::broadcast<bvw_t, W>(~(bvw_t)0);
  vacc acc;
  // The ensemble's base score is added on exactly one tile
  acc.from_vector(
      aie::broadcast<score_t, W>((score_t)(ADD_INIT ? bdtm::INIT_PREDICT : 0)));

#pragma unroll BDT_V2_UNROLL
  for (unsigned h = T_BEGIN; h < T_BEGIN + T_COUNT; h++) {
    vbvw v[BV_WORDS];
    if constexpr (BV_WORDS == 1) {
      v[0] = aie::broadcast<bvw_t, W>(bv_word(bdtm::INIT_V[h], 0));
    } else {
#pragma unroll
      for (unsigned q = 0; q < BV_WORDS; q++)
        v[q] = aie::broadcast<bvw_t, W>(bv_word(bdtm::INIT_V[h], q));
    }

    const unsigned b = h * bdtm::QS_NODES_PER_TREE;

    const auto thr = aie::load_unaligned_v<QT_CHUNK>(&QT_THR_P[b]);
    aie::vector<bvw_t, QT_CHUNK> bvv[BV_WORDS];
    bvv[0] = aie::load_unaligned_v<QT_CHUNK>(&QT_BV_P[b]);
    const auto bterm =
        aie::load_unaligned_v<TERM_LANES>(&bdtm::QT_BTERM[b * bdtm::MAX_TERMS]);
    const auto bsign =
        aie::load_unaligned_v<TERM_LANES>(&bdtm::QT_BSIGN[b * bdtm::MAX_TERMS]);
#pragma unroll
    for (unsigned j = 0; j < bdtm::QS_NODES_PER_TREE; j++) {
      vacc pa;
      pa.from_vector(aie::zeros<feat_t, W>());
#pragma unroll
      for (unsigned t = 0; t < bdtm::MAX_TERMS; t++) {
        const unsigned l = j * bdtm::MAX_TERMS + t;
        pa = aie::mac(pa, basis[bterm[l]], bsign[l]);
      }
      // No shift: the basis is already on the FX_SHIFT grid
      const vfeat p = pa.template to_vector<feat_t>(0);
      const vmask m = bdtm::SPLIT_LE ? aie::gt(p, thr[j]) : aie::ge(p, thr[j]);
      v[0] = aie::bit_and(v[0], aie::select(keep_all, bvv[0][j], m));
    }

    acc = aie::add(acc, leaf_value(v, h));
  }
  return acc.template to_vector<score_t>();
}

} // namespace bdtv
