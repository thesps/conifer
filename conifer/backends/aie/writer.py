import copy
import glob
import hashlib
import json
import logging
import math
import os
import shutil

import numpy as np

from conifer.backends.aie import checks, mapper, roles, tools
from conifer.backends.aie import shard as _shard
from conifer.backends.aie import tables as _tables
from conifer.backends.aie.devices import get_device_config
from conifer.backends.aie.platforms import find_platform, resolve_platform
from conifer.backends.aie.precision import COMPARE_WIDTH, SCORE_WIDTH, Precision
from conifer.backends.aie.report import read_aie_report
from conifer.backends.common import MultiPrecisionConfig
from conifer.model import ModelBase
from conifer.utils import copydocstring

logger = logging.getLogger(__name__)

AUTO = "auto"

DEFAULT_BATCH = 512

_DEFAULT_COMPARE = "ap_fixed<16,5,AP_RND_CONV,AP_SAT>"
_DEFAULT_SCORE = "ap_fixed<32,16,AP_RND_CONV,AP_SAT>"


class AIEConfig(MultiPrecisionConfig):
    backend = "aie"
    _config_fields = MultiPrecisionConfig._config_fields + [
        "priority",
        "n_tiles",
        "split_axis",
        "vector_width",
        "trees_per_tile",
        "n_samples",
        "shard",
        "feed",
        "plio_rate",
        "xilinx_part",
        "platform",
        "elfgen_jobs",
    ]
    _aie_alts = {  # noqa: RUF012
        "priority": ["Priority"],
        "n_tiles": ["NTiles"],
        "split_axis": ["SplitAxis"],
        "vector_width": ["VectorWidth", "W"],
        "trees_per_tile": ["TreesPerTile"],
        "n_samples": ["NSamples"],
        "shard": ["Shard"],
        "feed": ["Feed"],
        "plio_rate": ["PlioRate"],
        "xilinx_part": ["XilinxPart"],
        "platform": ["Platform"],
        "elfgen_jobs": ["ElfgenJobs"],
    }
    _alternates = {**MultiPrecisionConfig._alternates, **_aie_alts}  # noqa: RUF012
    _aie_defaults = {  # noqa: RUF012
        "precision": _DEFAULT_COMPARE,
        "score_precision": _DEFAULT_SCORE,
        "priority": "latency",
        "n_tiles": AUTO,
        "split_axis": AUTO,
        "vector_width": AUTO,
        "trees_per_tile": AUTO,
        "n_samples": AUTO,
        "shard": AUTO,
        "feed": AUTO,
        "plio_rate": AUTO,
        "xilinx_part": "xcve2802-vsvh1760-2MP-e-S",
        "platform": None,
        "elfgen_jobs": None,
    }
    _defaults = {**MultiPrecisionConfig._defaults, **_aie_defaults}  # noqa: RUF012
    _allow_undefined = [*MultiPrecisionConfig._allow_undefined] + [
        "platform",
        "elfgen_jobs",
    ]

    def __init__(self, configDict, validate=True):
        super().__init__(configDict, validate=False)
        for key, val in AIEConfig._aie_defaults.items():
            if getattr(self, key, None) is None and key in AIEConfig._config_fields:
                setattr(self, key, val)
        if self.score_precision is None or "ap_" not in str(self.score_precision):
            self.score_precision = _DEFAULT_SCORE
        if validate:
            self._validate()

    def default_config():
        return copy.deepcopy(AIEConfig._defaults)

    def _validate(self):
        super()._validate()
        self._extra_validate()

    def _extra_validate(self):
        if self.priority not in ("latency", "throughput"):
            raise ValueError(
                f"priority must be 'latency' or 'throughput', got '{self.priority}'"
            )
        if self.shard not in (AUTO, True, False, "false", "off"):
            raise ValueError(f"shard must be '{AUTO}' or False, got '{self.shard}'")
        if self.feed not in (AUTO, "memtile", "plio"):
            raise ValueError(
                f"feed must be 'memtile', 'plio' or '{AUTO}', got '{self.feed}'"
            )
        if self.split_axis not in (AUTO, "tree", "sample"):
            raise ValueError(
                f"split_axis must be 'tree', 'sample' or '{AUTO}', got "
                f"'{self.split_axis}'"
            )
        assert self.input_precision == self.threshold_precision, (
            f"input & threshold precision must be equal, got: {self.input_precision} & "
            f"{self.threshold_precision}"
        )


class AIEModel(ModelBase):
    def __init__(self, ensembleDict, config, metadata=None):
        super().__init__(ensembleDict, config, metadata)
        self.config = AIEConfig(config)
        cfg = self.config

        self.threshold_p = Precision(cfg.threshold_precision)
        self.score_p = Precision(cfg.score_precision)
        self.weight_p = Precision(cfg.weight_precision)
        self.threshold_p.validate("threshold", allowed=(COMPARE_WIDTH,))
        self.score_p.validate("score", allowed=(SCORE_WIDTH,))
        self.weight_p.validate("weight", allowed=(COMPARE_WIDTH,))

        checks.check_n_classes(self.n_classes)
        self.oblique = checks.check_weights(self.trees)
        self.device = get_device_config(cfg.xilinx_part)

        flat = [tc[0] for tc in self.trees]
        self.n_features_padded = checks.padded_n_features(self.n_features, self.oblique)
        if self.n_features_padded != self.n_features:
            logger.info(
                f"padding {self.n_features} features to {self.n_features_padded}: "
                f"the oblique kernel loads a weight row as one vector"
            )
            flat = [_pad_tree_features(t, self.n_features_padded) for t in flat]

        self.tables = _tables.QuickScorerTables(
            flat,
            self.n_features_padded,
            oblique=self.oblique,
            weight_precision=self.weight_p if self.oblique else None,
        )
        checks.check_max_depth(self.tables.max_depth)
        self.basis = self.tables.basis()

        self._resolve_mapping()
        self._check_shape()
        self._resolve_sharding()
        self.notes = self._notes

    def _resolve_sharding(self):
        """Shard the tables so each tile reads a contiguous window of feature rows

        Tree-split only: a sample-split tile holds the whole ensemble and reads every
        row, so there is nothing to cut.
        """
        self.sharding = None
        self.feed_memtile = False
        if self.config.shard in (False, "false", "off"):
            return
        if self.family != "axis" or self.split_axis != "tree" or self.n_tiles < 2:
            return
        # Sharding hands each tile a different row window, which only the memtile can
        # deliver: a multicast PLIO gives every tile the same stream
        if self.config.feed == "plio":
            logger.info(
                "feed=plio: not sharding, since only a memtile can hand each tile its own rows"
            )
            return
        self.sharding = _shard.Sharding(
            self.tables,
            self.n_trees_padded,
            self.n_features_padded,
            self.n_tiles,
            optimize="search",
        )
        self._verify_sharding()
        self.feed_memtile = True
        self._notes.append(
            f"sharded: each tile reads {self.sharding.max_rows_per_tile} of "
            f"{self.n_features_padded} feature rows at worst "
            f"({self.sharding.total_rows} across the array)"
            + (", fed from a memtile" if self.feed_memtile else "")
        )
        self._set_estimate()

    def _verify_sharding(self, n_samples=32, seed=0):
        """Require the sharded tables to score exactly what the unsharded ones do"""
        rng = np.random.default_rng(seed)
        hi = self.threshold_p.max_representable
        X = rng.uniform(-hi / 2, hi / 2, size=(n_samples, self.n_features))
        if self.n_features_padded != self.n_features:
            X = np.hstack(
                [X, np.zeros((n_samples, self.n_features_padded - self.n_features))]
            )
        bad = self.sharding.verify(
            X,
            self.threshold_p,
            self.score_p,
            self.init_predict[0],
            norm=self.norm,
            split_le=(self.splitting_convention == "<="),
        )
        if len(bad):
            raise RuntimeError(
                f"the sharded tables disagree with the unsharded ones on {len(bad)} of "
                f"{n_samples} samples"
            )

    def _resolve_plio_rate(self):
        """Offered PLIO rate in MHz"""
        cfg, dev = self.config, self.device
        if cfg.plio_rate == AUTO:
            return float(dev.get("plio_rate_mhz", 625))
        rate = float(cfg.plio_rate)
        # A PLIO runs at most half the AI Engine array clock
        ceiling = 1000.0 * dev["clock_ghz"] / 2
        if rate > ceiling:
            raise ValueError(
                f"plio_rate {rate} MHz exceeds {ceiling:g} MHz, half the "
                f"{dev['clock_ghz']} GHz array clock"
            )
        if rate <= 0:
            raise ValueError(f"plio_rate must be positive, got {rate}")
        return rate

    @property
    def feature_order(self):
        """Which global feature sits in each row of the input file"""
        return (
            self.sharding.fperm
            if self.sharding
            else list(range(self.n_features_padded))
        )

    @property
    def n_memtiles(self):
        return (self.n_tiles + 7) // 8 if self.feed_memtile else 0

    @property
    def family(self):
        return "oblique" if self.oblique else "axis"

    def _resolve_mapping(self):
        cfg = self.config
        d = self.tables.max_depth
        self._notes = []

        self.priority = cfg.priority
        self.split_axis = (
            ("tree" if self.priority == "latency" else "sample")
            if cfg.split_axis == AUTO
            else cfg.split_axis
        )
        # An oblique node reads a dense weight row, so there is no per-shard feature
        # frame to hand a tile
        want_feed = "plio" if (cfg.feed == "plio" or self.oblique) else "memtile"
        feed = want_feed if self.split_axis == "tree" else "plio"
        n, W, notes = mapper.choose_mapping(
            self.tables.n_trees,
            d,
            self.n_features_padded,
            self.threshold_p.n_bytes,
            self.score_p.n_bytes,
            self.priority,
            self.device["n_tiles"],
            self.device["clock_ghz"],
            self.oblique,
            feed,
            n_tiles=None if cfg.n_tiles == AUTO else int(cfg.n_tiles),
            W=None if cfg.vector_width == AUTO else int(cfg.vector_width),
            basis_n=self.basis["basis_n"] if self.oblique else 0,
        )
        self.n_tiles, self.W, self._notes = n, W, notes
        checks.check_vector_width(self.W, self.threshold_p.n_bytes)
        checks.check_n_tiles(
            self.n_tiles, self.device["n_tiles"], self.device.get("plio_channels_out")
        )

        self._finish_mapping(cfg)

    def _finish_mapping(self, cfg):
        """Everything that follows from (n_tiles, W, split_axis)"""
        if self.split_axis == "tree" and self.n_tiles > 1:
            self.trees_per_tile = (
                math.ceil(self.tables.n_trees / self.n_tiles)
                if cfg.trees_per_tile == AUTO
                else int(cfg.trees_per_tile)
            )
        else:
            self.trees_per_tile = self.tables.n_trees
        self.n_trees_padded = (
            self.trees_per_tile * self.n_tiles
            if self.split_axis == "tree"
            else self.tables.n_trees
        )
        if self.n_trees_padded > self.tables.n_trees:
            self._notes.append(
                f"padding {self.tables.n_trees} trees to {self.n_trees_padded} with null "
                f"trees so every tile takes exactly trees_per_tile={self.trees_per_tile}"
            )

        self.n_samples = (
            self._auto_n_samples()
            if cfg.n_samples == AUTO
            else self._round_batch(int(cfg.n_samples))
        )
        self.plio_rate = self._resolve_plio_rate()
        self._set_estimate()

    def _set_estimate(self):
        rows = (
            self.sharding.max_rows_per_tile
            if getattr(self, "sharding", None)
            else self.n_features_padded
        )
        self.estimate = mapper.estimate(
            self.n_trees_padded,
            self.tables.max_depth,
            rows,
            self.n_tiles,
            self.W,
            self.threshold_p.n_bytes,
            self.score_p.n_bytes,
            self.priority,
            self.device["clock_ghz"],
            self.oblique,
            "memtile" if getattr(self, "feed_memtile", False) else "plio",
            split_axis=self.split_axis,
            basis_n=self.basis["basis_n"] if self.oblique else 0,
        )

    def _leaf_bits(self, leaves_q):
        """Narrowest int that holds every quantized leaf

        The leaf select chain runs at the stored width, so storing leaves in the
        accumulator's width doubles every broadcast and select for nothing. Widening
        should happens once, at the accumulate.
        """
        peak = (
            int(np.max(np.abs(np.asarray(leaves_q, dtype=np.int64))))
            if len(leaves_q)
            else 0
        )
        for bits in (8, 16, 32):
            if peak <= 2 ** (bits - 1) - 1:
                return min(bits, self.score_p.width)
        return self.score_p.width

    def _report_leaf_width(self, leaves_q, bits):
        """Say when a wider binary point is costing the leaf select chain"""
        if bits <= 16 or self.score_p.width <= 16:
            return
        peak = int(np.max(np.abs(np.asarray(leaves_q, dtype=np.int64))))
        headroom = peak / (2**15 - 1)
        spare = self.score_p.shift - int(np.ceil(np.log2(headroom)))
        logger.info(
            f"leaves stored as {bits}-bit: the largest is {peak}, over the {2**15 - 1} a "
            f"16-bit leaf holds. ScorePrecision with {spare} fractional bits "
            f"(ap_fixed<{self.score_p.width},{self.score_p.width - spare}>) would fit"
        )

    @property
    def delta(self):
        """Samples a tile takes before the next tile's turn, under sample-split"""
        return self.W

    @property
    def batch_step(self):
        """Rows the graph must be compiled in whole multiples of"""
        return self.W * (self.n_tiles if self.split_axis == "sample" else 1)

    def _round_batch(self, n):
        """A run is a whole number of groups, so round up rather than refuse"""
        step = self.batch_step
        if n < 1:
            raise ValueError(f"n_samples must be at least 1, got {n}")
        rounded = step * math.ceil(n / step)
        if rounded != n:
            logger.info(
                f"n_samples {n} rounded up to {rounded}: a run is a whole number "
                f"of {step}-sample groups"
            )
        return rounded

    def _auto_n_samples(self):
        """Rows the graph is compiled for: a batch size, not a property of the model

        decision_function() pads a shorter X up to this and runs the graph repeatedly for
        a longer one, so it trades simulation time against per-run overhead, for more
        accurate measurements.
        """
        step = self.W * (self.n_tiles if self.split_axis == "sample" else 1)
        if self.split_axis == "sample" and step > DEFAULT_BATCH:
            self._notes.append(
                f"sample-split gives all {self.n_tiles} tiles a group of {self.W} at once, "
                f"so one run is {step} samples and the batch cannot be the {DEFAULT_BATCH} "
                f"a tree-split of this model would use. Scoring fewer than {step} rows "
                f"leaves some tiles with nothing but padding"
            )
        return step * math.ceil(DEFAULT_BATCH / step)

    def _check_shape(self):
        assert self.n_samples % self.batch_step == 0, "batch is not a whole run"
        p = self.tables.nodes_per_tree
        if self.oblique:
            terms = self.basis["max_terms"]
            if p * terms > 64:
                raise ValueError(
                    f"the oblique kernel loads one 64-lane vector of terms per tree, so "
                    f"QS_NODES_PER_TREE * MAX_TERMS must be <= 64, got {p} * {terms}. "
                    f"Reduce max_depth to {int(math.log2(64 // terms + 1))} or fewer"
                )
            # The kernel's working word follows MAX_LEAVES (uint16 or uint32), not the
            # generator's bv_t, which reaches uint64.
            bvw_bits = 16 if self.tables.max_leaves <= 16 else 32
            qt_lanes = max(1 << (p - 1).bit_length(), 8)
            if qt_lanes * bvw_bits > 1024:
                raise ValueError(
                    f"the oblique kernel has no chunked node loop, which max_depth "
                    f"{self.tables.max_depth} would need"
                )
        mem = mapper.table_bytes(
            self.n_trees_padded,
            self.tables.max_depth,
            self.tables.max_leaves,
            self.n_features_padded,
            self.W,
            self.threshold_p.n_bytes,
            self.score_p.n_bytes,
            self.oblique,
            self.basis["basis_n"] if self.oblique else 0,
            self.basis["max_terms"] if self.oblique else 1,
        )
        self.memory = mem
        checks.warn_tile_memory(mem, self.device["tile_memory_bytes"])
        checks.warn_score_range(
            self.n_trees_padded, float(np.max(np.abs(self.tables.leaves))), self.score_p
        )

    def resolved_config(self):
        """The configuration with every "auto" filled in, ready to be passed back"""
        cfg = self.config._to_dict()
        cfg.update(
            {
                "priority": self.priority,
                "n_tiles": self.n_tiles,
                "split_axis": self.split_axis,
                "vector_width": self.W,
                "trees_per_tile": self.trees_per_tile,
                "n_samples": self.n_samples,
                "shard": self.sharding is not None,
                "feed": "memtile" if self.feed_memtile else "plio",
                "plio_rate": self.plio_rate,
                "xilinx_part": self.config.xilinx_part,
                "platform": self.config.platform or self.device["platform"],
            }
        )
        return cfg

    @copydocstring(ModelBase.write)
    def write(self):
        cfg = self.config
        out = cfg.output_dir
        os.makedirs(f"{out}/src", exist_ok=True)
        os.makedirs(f"{out}/data", exist_ok=True)
        self.save()

        firmware = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firmware")
        for src in ("common", self.family):
            d = f"{firmware}/{src}"
            for name in sorted(os.listdir(d)):
                if os.path.isfile(f"{d}/{name}"):
                    shutil.copyfile(f"{d}/{name}", f"{out}/src/{name}")

        with open(f"{out}/src/parameters.h", "w") as f:
            f.write(self._parameters_h())
        # A tile's tree range is baked into its symbol, so the enumeration is
        # unavoidable
        with open(f"{out}/src/tile_roles.h", "w") as f:
            f.write(roles.tile_roles_h(self.n_tiles, self.split_axis, "plio"))
        with open(f"{out}/aie_model.json", "w") as f:
            json.dump(self._model_json(), f, indent=2)
        self._write_makefile()
        self._drop_stale_builds()

        for note in self._notes:
            logger.info(note)
        logger.info(
            f"estimated {self.estimate['est_cyc_per_sample']:.1f} cyc/sample, "
            f"latency_ss {self.estimate['est_latency_ss_ns']:.0f} ns on "
            f"{self.n_tiles} tile(s)"
        )

    def _sources_digest(self):
        """What the toolchain reads: everything under src/, and the Makefile"""
        out = self.config.output_dir
        h = hashlib.sha256()
        for path in sorted(glob.glob(f"{out}/src/*")) + [f"{out}/Makefile"]:
            h.update(os.path.basename(path).encode())
            with open(path, "rb") as f:
                h.update(f.read())
        return h.hexdigest()

    def _drop_stale_builds(self):
        """Remove a Work/ tree that was compiled from different sources

        aiecompiler keeps its state in <target>/Work and reuses it. Writing a new
        mapping into a directory that already holds one leaves the old per-core ELFs
        in place, but the kernel symbols change with the split, the tile count changes
        with the mapping, and what gets simulated is a mixture of the two.
        """
        out = self.config.output_dir
        stamp = f"{out}/.aie_sources"
        digest = self._sources_digest()
        previous = None
        if os.path.exists(stamp):
            with open(stamp) as f:
                previous = f.read().strip()
        if previous == digest:
            return
        for name in ("build_x86", "build_hw"):
            if os.path.isdir(f"{out}/{name}"):
                logger.info(
                    f"the generated sources changed: removing {name}, which "
                    f"aiecompiler would otherwise build on top of"
                )
                shutil.rmtree(f"{out}/{name}")
        with open(stamp, "w") as f:
            f.write(digest)

    def _write_makefile(self):
        cfg = self.config
        template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "template", "Makefile"
        )
        name = cfg.platform or self.device["platform"]
        platform = find_platform(name) or name
        subs = {
            "@PLATFORM@": platform,
            "@TOP@": "../src/tb.cpp",
            "@ELFGEN_JOBS@": "" if cfg.elfgen_jobs is None else str(cfg.elfgen_jobs),
            "@XIN@": f"{cfg.output_dir}/data/x.dat",
        }
        with open(template) as f:
            text = f.read()
        for k, v in subs.items():
            text = text.replace(k, v)
        with open(f"{cfg.output_dir}/Makefile", "w") as f:
            f.write(text)

    def _model_json(self):
        return {
            "family": self.family,
            "oblique": self.oblique,
            "n_trees": self.tables.n_trees,
            "n_trees_padded": self.n_trees_padded,
            "n_features": self.n_features,
            "n_features_padded": self.n_features_padded,
            "max_depth": self.tables.max_depth,
            "max_leaves": self.tables.max_leaves,
            "n_samples": self.n_samples,
            "fx_shift": self.threshold_p.shift,
            "val_shift": self.score_p.shift,
            "wgt_shift": self.weight_p.shift,
            "norm": self.norm,
            "config": self.resolved_config(),
            "estimate": self.estimate,
            "memory": self.memory,
            "sharding": None
            if self.sharding is None
            else {
                "n_feat": self.sharding.n_feat,
                "max_rows_per_tile": self.sharding.max_rows_per_tile,
                "total_rows": self.sharding.total_rows,
                "feature_order": self.sharding.fperm,
            },
            "n_memtiles": self.n_memtiles,
            "notes": self._notes,
        }

    def _parameters_h(self):
        t, cfg = self.tables, self.config
        T, P = self.n_trees_padded, t.nodes_per_tree
        L, F = t.max_leaves, self.n_features_padded
        bv_c = {16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}[t.bv_bits]
        q = t.quantize(self.threshold_p, self.score_p, self.norm)
        init_q = int(
            self.score_p.quantize([float(self.init_predict[0]) * self.norm])[0]
        )
        all_ones = (1 << t.bv_bits) - 1

        sh = self.sharding
        if sh is None:
            qt_feat = _pad(t.qt_group, T * P, 0)
            qt_thr = _pad(list(q["qt_thr"]), T * P, 0)
            qt_bv = _pad(list(t.qt_bv), T * P, all_ones)
            init_v = _pad(list(t.init_v), T, all_ones)
            leaves = np.zeros((T, L), dtype=np.int64)
            leaves[: t.n_trees] = q["leaves"]
        else:
            qt_feat = list(sh.qt_group)
            qt_thr = list(self.threshold_p.quantize(np.asarray(sh.qt_thr_f)))
            qt_bv = list(sh.qt_bv)
            init_v = list(sh.init_v)
            leaves = self.score_p.quantize(sh.leaves * self.norm)

        leaf_bits = self._leaf_bits(leaves.ravel())
        self._report_leaf_width(leaves.ravel(), leaf_bits)

        s = [
            "// Generated by conifer's AIE backend. Do not edit.",
            "#pragma once",
            "#include <cstdint>",
            "",
        ]
        s.append(f"#define BDT_W {self.W}")
        s.append(f"#define BDT_PLIO_RATE {self.plio_rate}")
        s.append(f'#define XIN_FILE "{cfg.output_dir}/data/x.dat"')
        s += [
            f"#define BDT_N_TILES {self.n_tiles}",
            f"#define BDT_SPLIT_TREE {1 if self.split_axis == 'tree' else 0}",
            f"#define BDT_DELTA {self.delta}",
        ]
        if self.family == "axis":
            s += [
                "#define BDT_FEED_PLIO 0",
                f"#define BDT_TAU {self.trees_per_tile if self.split_axis == 'tree' else 0}",
                f"#define BDT_SHARDED {1 if self.sharding else 0}",
                f"#define BDT_FEED_MEMTILE {1 if self.feed_memtile else 0}",
                "#define BDT_MT_FANOUT 8",
                "#define BDT_MT_BUFFERS 2",
            ]
        s += [
            "",
            "namespace bdtm {",
            "",
            f"typedef {self.threshold_p.c_type} feat_t;",
            f"typedef {self.score_p.c_type} score_t;",
            f"typedef {_int_c_type(leaf_bits)} leaf_t;",
            f"typedef {bv_c} bv_t;",
            "",
            f"constexpr unsigned N_FEATURES = {F};",
            f"constexpr unsigned N_TREES    = {T};",
            f"constexpr unsigned MAX_LEAVES = {L};",
            f"constexpr unsigned N_SAMPLES  = {self.n_samples};",
            f"constexpr unsigned QS_NODES_PER_TREE = {P};",
            f"constexpr bool SPLIT_LE = {'true' if self.splitting_convention == '<=' else 'false'};",
            f"constexpr score_t INIT_PREDICT = {init_q};",
        ]
        if self.oblique:
            b = self.basis
            s += [
                f"constexpr int WGT_SHIFT = {self.weight_p.shift};",
                f"constexpr unsigned BASIS_N = {b['basis_n']};",
                f"constexpr unsigned MAX_TERMS = {b['max_terms']};",
            ]
        s.append("")
        if sh is not None:
            s += [
                "}  // namespace bdtm",
                "",
                "namespace bdtsh {",
                "",
                f"constexpr unsigned N_SHARDS = {self.n_tiles};",
                "constexpr bool WINDOWED = true;",
                _array("unsigned", "T_BEGIN", sh.t_begin),
                _array("unsigned", "T_COUNT", sh.t_count),
                _array("unsigned", "N_FEAT", sh.n_feat),
                _array("unsigned", "OFFSET", sh.offset),
                "}  // namespace bdtsh",
                "",
                "namespace bdtm {",
                "",
            ]
        s.append(_array("int16_t", "QT_FEAT", qt_feat))
        s.append(_array(self.threshold_p.c_type, "QT_THR", qt_thr))
        s.append(_array("bv_t", "QT_BV", qt_bv, hexw=t.bv_bits))
        s.append(_array("bv_t", "INIT_V", init_v, hexw=t.bv_bits))
        s.append(_array("leaf_t", "LEAVES", leaves.ravel()))
        if self.oblique:
            b = self.basis
            n = T * P * b["max_terms"] + 64
            s.append(_array("int16_t", "BASIS_I", b["basis_i"]))
            s.append(_array("int16_t", "BASIS_J", b["basis_j"]))
            s.append(_array(self.threshold_p.c_type, "BASIS_WI", b["basis_wi"]))
            s.append(_array(self.threshold_p.c_type, "BASIS_WJ", b["basis_wj"]))
            s.append(_array("int16_t", "QT_BTERM", _pad(list(b["qt_bterm"]), n, 0)))
            s.append(
                _array(
                    self.threshold_p.c_type, "QT_BSIGN", _pad(list(b["qt_bsign"]), n, 0)
                )
            )
        s += ["}  // namespace bdtm", ""]
        return "\n".join(s)

    def platform(self):
        """Absolute path of the .xpfm this project builds against"""
        return resolve_platform(self.config.platform or self.device["platform"])

    @copydocstring(ModelBase.compile)
    def compile(self):
        self.write()
        return tools.run_make(
            self.config.output_dir, "x86sim_build", PLATFORM=self.platform()
        )

    @copydocstring(ModelBase.decision_function)
    def decision_function(self, X, trees=False):
        if trees:
            logger.warning(
                "Individual tree output (trees=True) is not implemented for the aie "
                "backend"
            )
        X = np.asarray(X)
        assert X.shape[1] == self.n_features, (
            f"Wrong number of features, expected {self.n_features}, got {X.shape[1]}"
        )
        n, batch = len(X), self.n_samples
        runs = math.ceil(n / batch)
        if runs > 1:
            logger.info(
                f"scoring {n} samples in {runs} runs of {batch}: the graph is "
                f"compiled for a fixed batch, set NSamples to change it"
            )
        elif n < batch:
            msg = (
                f"scoring {n} samples in a graph compiled for {batch}: the "
                f"{batch - n} padding rows are computed and discarded, set "
                f"NSamples to trim them"
            )
            # Under sample-split the padding is not spread evenly: groups are dealt to
            # tiles in turn, so a short X starves the tiles at the end of the round.
            if self.split_axis == "sample" and self.n_tiles > 1:
                idle = self.n_tiles - math.ceil(n / self.W)
                if idle > 0:
                    msg += (
                        f". {idle} of the {self.n_tiles} tiles score only padding: "
                        f"{n} rows fill {math.ceil(n / self.W)} of the {self.n_tiles} "
                        f"groups a run deals"
                    )
            logger.info(msg)
        out = []
        for i in range(runs):
            self.write_input(X[i * batch : (i + 1) * batch])
            if not tools.run_make(
                self.config.output_dir, "x86sim", PLATFORM=self.platform()
            ):
                return None
            out.append(self.read_scores()[:batch])
        return np.concatenate(out)[:n] if out else np.empty(0)

    def write_input(self, X):
        """Feature-major, W-blocked, four values a line for the 64-bit PLIO"""
        cfg = self.config
        X = np.asarray(X, dtype=np.float64)
        if len(X) > self.n_samples:
            raise ValueError(
                f"{len(X)} rows exceeds the {self.n_samples} this graph runs; "
                f"decision_function splits a longer X into runs"
            )
        if len(X) < self.n_samples:
            X = np.vstack([X, np.zeros((self.n_samples - len(X), X.shape[1]))])
        if X.shape[1] != self.n_features_padded:
            X = np.hstack([X, np.zeros((len(X), self.n_features_padded - X.shape[1]))])
        xq = self.threshold_p.quantize(X)
        toks = []
        for g in range(0, len(xq), self.W):
            block = xq[g : g + self.W]
            for k in self.feature_order:
                toks.extend(int(v) for v in block[:, k])
        per_line = 4
        os.makedirs(f"{cfg.output_dir}/data", exist_ok=True)
        lines = [
            " ".join(str(v) for v in toks[i : i + per_line])
            for i in range(0, len(toks), per_line)
        ]
        base = f"{cfg.output_dir}/data/x.dat"
        for path, chunk in self._input_files(base, lines):
            with open(path, "w") as f:
                f.write("\n".join(chunk) + "\n")

    def _input_files(self, base, lines):
        """One file, or one per tile when each scores its own samples

        Sample-split deals whole turns of samples round-robin, so each tile reads its
        own cut. The name carries the mapping, as the graph builds it.
        """
        if self.feed_memtile or self.split_axis == "tree" or self.n_tiles == 1:
            return [(base, lines)]
        lines_per_group = self.n_features_padded * self.W // 4
        per_turn = lines_per_group * max(1, self.delta // self.W)
        out = [[] for _ in range(self.n_tiles)]
        for i in range(0, len(lines), per_turn):
            out[(i // per_turn) % self.n_tiles].extend(lines[i : i + per_turn])
        root, ext = os.path.splitext(base)
        key = f".n{self.n_tiles}d{self.delta}"
        return [(f"{root}{key}.t{t}{ext}", out[t]) for t in range(self.n_tiles)]

    @property
    def n_outputs(self):
        """Output ports the graph declares: every tile emits its own partial"""
        return self.n_tiles

    def _score_dir(self, simulator="x86"):
        order = (
            ("build_hw/aiesimulator_output",)
            if simulator == "aie"
            else (
                "build_x86/x86simulator_output",
                "build_hw/aiesimulator_output",
                "x86simulator_output",
                "data",
            )
        )
        for d in order:
            p = f"{self.config.output_dir}/{d}"
            if os.path.exists(f"{p}/scores.dat"):
                return p
        raise FileNotFoundError(f"No scores.dat under {self.config.output_dir}")

    def _read_one(self, path):
        vals = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("T"):
                    continue
                vals.extend(int(v) for v in line.split())
        return np.asarray(vals, dtype=np.int64)

    def read_scores(self, filename=None, simulator="x86"):
        """Integer scores from the simulator output, dequantized

        Tree-split emits one partial score per tile, which sum to the ensemble score,
        sample-split emits whole scores interleaved in turns of W.

        simulator picks whose output to read: 'x86' is the run decision_function made,
        'aie' the cycle-accurate one build() made. They score the same input and should
        agree exactly.
        """
        if filename is not None:
            return self.score_p.dequantize(self._read_one(filename))
        d = self._score_dir(simulator)
        parts = [
            self._read_one(f"{d}/scores.dat" if i == 0 else f"{d}/scores.t{i}.dat")
            for i in range(self.n_outputs)
        ]
        if len(parts) == 1:
            return self.score_p.dequantize(parts[0])
        n = min(len(p) for p in parts)
        parts = [p[:n] for p in parts]
        if self.split_axis == "tree":
            return self.score_p.dequantize(np.sum(parts, axis=0))
        out = np.empty(n * len(parts), dtype=np.int64)
        for i, p in enumerate(parts):
            for g in range(0, n, self.W):
                out[
                    i * self.W + g * len(parts) : i * self.W + g * len(parts) + self.W
                ] = p[g : g + self.W]
        return self.score_p.dequantize(out)

    def build(self, X=None, simulate=True, **kwargs):
        """Compile the mapping for hardware, and time it

        Two toolchain stages, and which of them to run is what the arguments say.
        aiecompiler places the design and reports what it costs, aiesimulator runs it
        cycle-accurately, which needs a stimulus and takes roughly twice as long.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features), optional
            Rows for the cycle-accurate run to score. Read back with
            read_scores(simulator='aie'). Defaults to whatever data/x.dat holds, or
            zeros if nothing has been written, which is fine for timing, as the kernels
            spend the same cycles on whatever the data.

        simulate: bool, optional
            Run aiesimulator. False stops after the hardware compile, which gives the
            placement, tile memory and program memory without the simulator's time.
            Defaults to True.

        Returns
        ----------
        success: bool
        """
        if X is not None and not simulate:
            raise ValueError(
                "build(X=...) names rows to score but simulate=False "
                "runs nothing that would score them"
            )
        self.write()
        if simulate:
            self._stimulus(X)
        return tools.run_make(
            self.config.output_dir,
            "aiesim" if simulate else "hw_build",
            PLATFORM=self.platform(),
        )

    def _stimulus(self, X):
        """What the cycle-accurate run will score

        The simulator reads its input from a file, and reports a missing one only after
        it has compiled the whole graph.
        """
        path = f"{self.config.output_dir}/data/x.dat"
        if X is not None:
            self.write_input(X)
        elif os.path.exists(path):
            logger.info(
                f"simulating the rows already in {path}. Pass build(X) to specify "
                f"them, since decision_function() leaves its last batch here"
            )
        else:
            logger.info(
                f"no input written yet: simulating {self.n_samples} rows of zeros "
                f"for timing. Pass build(X) for scores"
            )
            self.write_input(np.zeros((self.n_samples, self.n_features)))

    def read_report(self) -> dict:
        """Read whatever stage of report is on disk

        Returns
        ----------
        dictionary of extracted report contents, with a 'stage' key naming what was found
        """
        return read_aie_report(self.config.output_dir)


_INT_C = {8: "int8_t", 16: "int16_t", 32: "int32_t"}


def _int_c_type(bits):
    return _INT_C[bits]


def _pad(values, n, fill):
    values = list(values)
    return values + [fill] * (n - len(values))


def _array(ctype, name, values, hexw=None):
    values = list(values)
    if hexw:
        digits = hexw // 4
        suffix = "ull" if hexw == 64 else "u"
        toks = [f"0x{int(v) & ((1 << hexw) - 1):0{digits}x}{suffix}" for v in values]
    else:
        toks = [str(int(v)) for v in values]
    lines = [f"constexpr {ctype} {name}[{len(toks)}] = {{"]
    for i in range(0, len(toks), 12):
        lines.append("  " + ", ".join(toks[i : i + 12]) + ",")
    lines.append("};\n")
    return "\n".join(lines)


def _pad_tree_features(tree, n_features):
    """Widen every weight row with zero columns, which no split references"""
    t = copy.copy(tree)
    rows = np.asarray(tree.weight, dtype=np.float64)
    wide = np.zeros((rows.shape[0], n_features))
    wide[:, : rows.shape[1]] = rows
    t.weight = wide.tolist()
    return t


def make_model(ensembleDict, config):
    return AIEModel(ensembleDict, config)


def auto_config(granularity="simple"):
    """Create an initial configuration for the aie backend"""
    config = {
        "Backend": "aie",
        "ProjectName": "my_prj",
        "OutputDir": "my-conifer-prj",
        "Precision": _DEFAULT_COMPARE,
        "ScorePrecision": _DEFAULT_SCORE,
        "Priority": "latency",
        "NTiles": AUTO,
        "SplitAxis": AUTO,
        "VectorWidth": AUTO,
        "TreesPerTile": AUTO,
        "NSamples": AUTO,
        "Shard": AUTO,
        "Feed": AUTO,
        "PlioRate": AUTO,
        "XilinxPart": "xcve2802-vsvh1760-2MP-e-S",
    }
    return config
