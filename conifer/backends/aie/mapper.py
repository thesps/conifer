import logging
import math

logger = logging.getLogger(__name__)

VECTOR_WIDTHS = (8, 16, 32, 64)

# A memtile row is two 256-bit loads against sixteen 32-bit stream reads
_MEMTILE_ROW_FRACTION = 0.125

# The width the cost table was fitted at, and one vector register
_FIT_W = 32
_FIT_BYTES = 64
_REGISTER_BITS = 512

# The share of per-tree work that does not scale with the bitvector's register count
_PER_TREE_FLOOR = 0.5

# Measured: fixed + 11.9 * basis_n + 380 * trees_per_tile
_OBLIQUE_PER_TREE_TAX = 380.0 / 156.0
_BASIS_CYC_PER_ENTRY = 11.9

# cyc/invocation = base + n_features * row + per_tree * trees_per_tile, measured at f16/W=32/int16
# Piecewise in depth as the cost steps where bv_t widens
_DEPTH_COST = {
    1: (106, 35.4),
    2: (107, 50.5),
    3: (104, 91.0),
    4: (106, 156.0),
    5: (125, 410.4),
    6: (125, 1150.5),
}

# An AIE-ML vector register is 512, with optimized hardware pairing to use 512x2=1024 bit data types
_PRIORITY_REGISTER_BITS = {"latency": 512, "throughput": 1024}

# The narrowest group the study swept
MIN_VECTOR_WIDTH = 8

# Auto stops here for compile time, but both priorities keep paying past this
_AUTO_TILE_CEILING = 16


def tile_candidates(ceiling):
    """Powers of two up to a ceiling."""
    out, n = [], 1
    while n <= ceiling:
        out.append(n)
        n *= 2
    return out


def bitvector_bits(max_depth):
    """Bits of result bitvector per lane: one bit per leaf, rounded to the word type"""
    leaves = 1 << max_depth
    if leaves <= 16:
        return 16
    if leaves <= 32:
        return 32
    return 64


def lane_bits(max_depth, feat_bytes):
    """Bits per sample lane in the per-node loop

    A node touches two vectors, both W lanes wide: the result bitvector, one bit per
    leaf, and the feature row it tests, one feat_t per lane. The wider of the two sets
    the loop's register pressure.

    The accumulator and the leaf select are wider per lane but run once per tree, so
    they do not set the width.
    """
    return max(bitvector_bits(max_depth), 8 * feat_bytes)


def vector_width(priority, max_depth, feat_bytes):
    """Samples per kernel invocation: the group whose inner-loop vector fills the register

    Latency fills the 512-bit register and throughput 2x512-bit ones. Note that a deeper
    ensemble carries more leaf bits, so a narrower group fills the same register.

    Example measurement on one tile, int16, t32-f16, aiesimulator:
        depth 4 (16-bit lane)   latency_ss  W=16 4316.0 ns   W=32  4291.8 ns
        depth 5 (32-bit lane)   latency_ss  W=16 7978.9 ns   W=32 10816.0 ns

    Analogously, W=64 at depth 4 gives better throughput by 12.6% vs W=32
    """
    W = _PRIORITY_REGISTER_BITS[priority] // lane_bits(max_depth, feat_bytes)
    return min(max(W, MIN_VECTOR_WIDTH), 1024 // (8 * feat_bytes))


def _metric(priority):
    return (
        "est_latency_ss_ns" if priority == "latency" else "est_throughput_ns_per_sample"
    )


def choose_mapping(
    n_trees,
    max_depth,
    n_features,
    feat_bytes,
    leaf_bytes,
    priority,
    max_tiles,
    clock_ghz,
    oblique=False,
    feed="plio",
    n_tiles=None,
    W=None,
    basis_n=0,
):
    """Pick the tile count, on the metric the priority names. W follows the priority."""
    notes = []
    ceiling = max_tiles
    auto_ceiling = min(ceiling, _AUTO_TILE_CEILING)

    tiles = [n_tiles] if n_tiles else tile_candidates(auto_ceiling)
    # W is not searched. Follows from the register the priority wants to fill.
    w = W or vector_width(priority, max_depth, feat_bytes)

    key = _metric(priority)
    best, best_score = None, None
    for n in tiles:
        e = estimate(
            n_trees,
            max_depth,
            n_features,
            n,
            w,
            feat_bytes,
            leaf_bytes,
            priority,
            clock_ghz,
            oblique,
            feed,
            basis_n=basis_n,
        )
        if best_score is None or e[key] < best_score:
            best, best_score = n, e[key]

    n = best
    if n_tiles is None and n == auto_ceiling < ceiling:
        notes.append(
            f"stopping at {n} tiles! auto does not go past {auto_ceiling} to keep "
            f"compile time down, raise n_tiles for more"
        )
    if oblique:
        notes.append(
            "oblique: tree-split does not divide the basis, so the speedup "
            "saturates against it however many tiles are added"
        )
    return n, w, notes


def _row_cycles(W, feat_bytes):
    # A core has one 32-bit input stream, so a row of W * sizeof(feat_t) bytes arrives
    # four at a time, one read per cycle
    return W * feat_bytes / 4


def _bitvector_registers(W, max_depth):
    return max(1, math.ceil(bitvector_bits(max_depth) * W / _REGISTER_BITS))


def _register_scale(W, max_depth):
    """Part of the per-tree work depends on bitvector width."""
    ratio = _bitvector_registers(W, max_depth) / _bitvector_registers(_FIT_W, max_depth)
    return _PER_TREE_FLOOR + (1.0 - _PER_TREE_FLOOR) * ratio


def _invocation_parts(
    n_features, max_depth, trees_per_tile, W, feat_bytes, feed="plio"
):
    """Invocation costs per tree.

    The law is `base + n_features * row + per_tree * trees_per_tile` (per-invocation
    constant + cost of reading the feature rows + tree work).

    n_features is the busiest tile's row count, which sharding reduces. A memtile row is
    two wide loads rather than sixteen stream reads, so `n_features * row` costs less
    using that feed option.
    """
    base, per_tree = _DEPTH_COST[max_depth]
    row = _row_cycles(W, feat_bytes) * (
        _MEMTILE_ROW_FRACTION if feed == "memtile" else 1.0
    )
    # base scales with the group, like the rows do: measured, it roughly halves from
    # W=32 to W=16
    base *= W * feat_bytes / _FIT_BYTES
    # first two independepent on trees_per_tile
    return base + n_features * row, per_tree * _register_scale(
        W, max_depth
    ) * trees_per_tile


def invocation_cycles(
    n_features, max_depth, trees_per_tile, W, feat_bytes, feed="plio"
):
    """Cycles for one invocation, which scores W samples against trees_per_tile trees"""
    fixed, trees = _invocation_parts(
        n_features, max_depth, trees_per_tile, W, feat_bytes, feed
    )
    return fixed + trees


def _roundup(x, to):
    return ((x + to - 1) // to) * to


def table_bytes(
    n_trees,
    max_depth,
    max_leaves,
    n_features,
    W,
    feat_bytes,
    leaf_bytes,
    oblique,
    basis_n=0,
    max_terms=1,
):
    """Tile data memory, mirroring the heap and stack the graph declares

    Every tile is handed the same generated tables and indexes its own tree range into
    them, so this does not shrink with the tile count. Sharding cuts on data movement
    (tiles reads less feature rows), but not the data structures' size.
    """
    slots = n_trees * ((1 << max_depth) - 1)
    bv_bytes = bitvector_bits(max_depth) // 8
    tables = {
        "qt_thr": slots * feat_bytes,
        "qt_bv": slots * bv_bytes,
        "qt_feat": slots * 2,
        "leaves": n_trees * max_leaves * leaf_bytes,
        "init_v": n_trees * bv_bytes,
    }
    x_lanes = n_features
    if oblique:
        tables["qt_bterm"] = (slots * max_terms + 64) * 2
        tables["qt_bsign"] = (slots * max_terms + 64) * feat_bytes
        tables["basis_ij"] = basis_n * 2 * 2
        tables["basis_w"] = basis_n * 2 * feat_bytes
        x_lanes = n_features + basis_n

    b = dict(tables)
    b["heap"] = _roundup(sum(tables.values()) + 2 * 1024, 1024)
    b["stack"] = _roundup(x_lanes * W * feat_bytes + 4 * 1024, 1024)
    b["total"] = b["heap"] + b["stack"]
    return b


def _split_axis(priority):
    return "tree" if priority == "latency" else "sample"


def _trees_per_tile_for(n_trees, n_tiles, split_axis):
    # Sample-split gives every tile the whole ensemble, tree-split divides it, with the
    # busiest tile taking the ceiling
    if split_axis == "sample":
        return n_trees
    return math.ceil(n_trees / n_tiles)


def estimate(
    n_trees,
    max_depth,
    n_features,
    n_tiles,
    W,
    feat_bytes,
    leaf_bytes,
    priority,
    clock_ghz,
    oblique=False,
    feed="plio",
    split_axis=None,
    basis_n=0,
):
    """Forward estimate for one mapping"""
    split_axis = split_axis or _split_axis(priority)
    trees_per_tile = _trees_per_tile_for(n_trees, n_tiles, split_axis)
    fixed, trees = _invocation_parts(
        n_features, max_depth, trees_per_tile, W, feat_bytes, feed
    )
    if oblique:
        inv = fixed + _OBLIQUE_PER_TREE_TAX * trees + _BASIS_CYC_PER_ENTRY * basis_n
    else:
        inv = fixed + trees
    samples_per_inv = W * (n_tiles if split_axis == "sample" else 1)

    return {
        "est_cyc_per_invocation": inv,
        "est_cyc_per_sample": inv / samples_per_inv,
        "est_latency_ss_ns": 1.1 * inv / clock_ghz,
        "est_throughput_ns_per_sample": inv / samples_per_inv / clock_ghz,
        "trees_per_tile": trees_per_tile,
        "split_axis": split_axis,
    }
