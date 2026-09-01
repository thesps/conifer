#pragma once

#include "parameters.h"
#include <cstdint>

#ifndef BDT_SHARDED
#define BDT_SHARDED 0
#endif

#ifndef BDT_W
#define BDT_W 16
#endif

#ifndef BDT_N_TILES
#define BDT_N_TILES 1
#endif

// 1 = tree-split   : disjoint tree subsets, same samples, partial scores merged
// 0 = sample-split : full ensemble per tile, distinct samples, no merge
#ifndef BDT_SPLIT_TREE
#define BDT_SPLIT_TREE 0
#endif

#ifndef BDT_FEED_PLIO
#define BDT_FEED_PLIO 0
#endif

#ifndef BDT_FEED_MEMTILE
#define BDT_FEED_MEMTILE 0
#endif

#ifndef BDT_MT_BUFFERS
#define BDT_MT_BUFFERS 2
#endif

#ifndef BDT_MT_FANOUT
#define BDT_MT_FANOUT 8
#endif

#ifndef BDT_DELTA
#define BDT_DELTA BDT_W
#endif

#ifndef BDT_PLIO_RATE
#define BDT_PLIO_RATE 625
#endif

#ifndef BDT_TAU
#define BDT_TAU 0 /* 0 = derive from N_TILES */
#endif

namespace bdtmt {

constexpr unsigned N_TILES = BDT_N_TILES;
constexpr bool SPLIT_TREE = (BDT_SPLIT_TREE != 0);

// Sample-split gives every tile the whole ensemble, only tree-split shards it.
constexpr unsigned N_SHARDS = SPLIT_TREE ? N_TILES : 1u;
constexpr unsigned DELTA = BDT_DELTA;
constexpr bool FEED_PLIO = (BDT_FEED_PLIO != 0);
constexpr bool FEED_MEMTILE = (BDT_FEED_MEMTILE != 0);
constexpr unsigned MT_BUFFERS = BDT_MT_BUFFERS;
static_assert(MT_BUFFERS >= 2, "a ping-pong needs at least two buffers");
constexpr unsigned MT_FANOUT = BDT_MT_FANOUT;
static_assert(MT_FANOUT >= 1, "a memtile feeds at least one tile");
// How many memtiles the array needs, which tile each one feeds, and how many it
// feeds.
constexpr unsigned N_MEMTILE =
    FEED_MEMTILE ? (N_TILES + MT_FANOUT - 1) / MT_FANOUT : 0u;
constexpr unsigned mt_of(unsigned t) { return t / MT_FANOUT; }
constexpr unsigned mt_first(unsigned m) { return m * MT_FANOUT; }
constexpr unsigned mt_count(unsigned m) {
  return (m + 1) * MT_FANOUT <= N_TILES ? MT_FANOUT : N_TILES - m * MT_FANOUT;
}
static_assert(
    !(FEED_MEMTILE && FEED_PLIO),
    "broadcast, per-tile PLIO and memtile are three feeds, not two flags");

#if BDT_FEED_MEMTILE
static_assert(SPLIT_TREE && N_TILES > 1,
              "the memtile feed writes one group for the whole array to share, "
              "a sample-split tile holds different samples and shares nothing");
static_assert(BDT_SHARDED, "FEED=memtile hands a tile a range of rows, so the "
                           "model must have been sharded");
static_assert(bdtsh::WINDOWED,
              "sharding's rows need to be contiguous for memtile");
#endif

constexpr double PLIO_RATE = BDT_PLIO_RATE; // MHz, on every input port

static_assert(
    BDT_SHARDED || BDT_TAU != 0 || bdtm::N_TREES % N_SHARDS == 0,
    "n_trees must divide evenly across the tiles, except for sharded models");
constexpr unsigned TAU =
    (BDT_TAU != 0) ? (unsigned)BDT_TAU : bdtm::N_TREES / N_SHARDS;
static_assert(TAU <= bdtm::N_TREES, "tau exceeds the ensemble");

constexpr bool SHARDED = (BDT_SHARDED != 0);

#if BDT_SHARDED
static_assert(bdtsh::N_SHARDS == N_SHARDS,
              "this model was sharded for a different tile count");
static_assert(SPLIT_TREE || N_TILES == 1,
              "sharding is a property of the tree axis. a sample-split tile "
              "holds the whole ensemble and therefore reads every feature");
#endif

constexpr unsigned t_begin(unsigned shard) {
#if BDT_SHARDED
  return bdtsh::T_BEGIN[shard];
#else
  return shard * TAU;
#endif
}
constexpr unsigned t_count(unsigned shard) {
#if BDT_SHARDED
  return bdtsh::T_COUNT[shard];
#else
  (void)shard;
  return TAU;
#endif
}

constexpr unsigned n_feat(unsigned shard) {
#if BDT_SHARDED
  return bdtsh::N_FEAT[shard];
#else
  (void)shard;
  return bdtm::N_FEATURES;
#endif
}

constexpr unsigned feat_offset(unsigned shard) {
#if BDT_SHARDED
  return bdtsh::OFFSET[shard];
#else
  (void)shard;
  return 0u;
#endif
}

constexpr bool adds_init(unsigned shard) { return shard == 0; }

} // namespace bdtmt

static_assert(bdtm::N_SAMPLES % BDT_W == 0,
              "N_SAMPLES must be a whole number of W-sample groups");
static_assert(bdtmt::SPLIT_TREE ||
                  (bdtm::N_SAMPLES / BDT_W) % bdtmt::N_TILES == 0,
              "sample-split needs the group count to divide across the tiles");
constexpr unsigned iter_count =
    bdtmt::SPLIT_TREE ? (bdtm::N_SAMPLES / BDT_W)
                      : (bdtm::N_SAMPLES / BDT_W / bdtmt::N_TILES);
