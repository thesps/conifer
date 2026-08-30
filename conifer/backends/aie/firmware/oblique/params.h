#pragma once

#include "parameters.h"
#include <cstdint>

#ifndef BDT_W
#define BDT_W 16
#endif

#ifndef BDT_N_TILES
#define BDT_N_TILES 1
#endif

// 1 = tree-split   : disjoint tree subsets, same samples, partial scores summed
// off-array 0 = sample-split : full ensemble per tile, distinct samples, no
// merge
#ifndef BDT_SPLIT_TREE
#define BDT_SPLIT_TREE 0
#endif

// Samples a tile takes before the next tile's turn, under sample-split
#ifndef BDT_DELTA
#define BDT_DELTA BDT_W
#endif

#ifndef BDT_PLIO_RATE
#define BDT_PLIO_RATE 625
#endif

namespace bdtmt {

constexpr unsigned N_TILES = BDT_N_TILES;
constexpr bool SPLIT_TREE = (BDT_SPLIT_TREE != 0);
constexpr unsigned N_SHARDS = SPLIT_TREE ? N_TILES : 1u;
constexpr unsigned DELTA = BDT_DELTA;
constexpr double PLIO_RATE = BDT_PLIO_RATE;

constexpr unsigned TAU = bdtm::N_TREES / N_SHARDS;
static_assert(bdtm::N_TREES % N_SHARDS == 0,
              "n_trees must divide evenly across the tiles");

constexpr unsigned t_begin(unsigned shard) { return shard * TAU; }
constexpr unsigned t_count(unsigned) { return TAU; }

// The ensemble's base score belongs to exactly one tile.
constexpr bool adds_init(unsigned shard) { return shard == 0; }

} // namespace bdtmt

// One group of W samples per kernel invocation, so the graph runs N_SAMPLES / W
// times and the profile's cycles/call covers W samples.
static_assert(bdtm::N_SAMPLES % BDT_W == 0,
              "N_SAMPLES must be a whole number of W-sample groups");
static_assert(bdtmt::SPLIT_TREE ||
                  (bdtm::N_SAMPLES / BDT_W) % bdtmt::N_TILES == 0,
              "sample-split needs the group count to divide across the tiles");
constexpr unsigned iter_count =
    bdtmt::SPLIT_TREE ? (bdtm::N_SAMPLES / BDT_W)
                      : (bdtm::N_SAMPLES / BDT_W / bdtmt::N_TILES);
