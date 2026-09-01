# AI Engine backend

Compiles a trained BDT to an AMD AI Engine project, using vectorized branch-free kernels mapped across one or more AIE tiles.

Main tested target: VEK280 (`xcve2802`), AIE-MLv1, on Vitis 2026.1.

Source `/path/to/Vitis/settings64.sh` to set up the Vitis environment. `PLATFORM_REPO_PATHS` can optionally be set. Both the `vek280_base` and `xilinx_vek280_base_<version>` layouts are detected. Set `Platform` to an `.xpfm` path to override.

## Quick start

```python
import conifer

cfg = conifer.backends.aie.auto_config()
cfg["OutputDir"] = "prj_aie"
cfg["Priority"] = "latency"  # or 'throughput'

model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()  # no toolchain needed
model.compile()  # aiecompiler
y = model.decision_function(X)  # x86simulator
model.build()  # aiecompiler + aiesimulator
print(model.read_report())
```

Take a look at `examples/sklearn_to_aie.py` and `examples/ydf_to_aie.py`!

## Backend map

    writer.py     AIEConfig, AIEModel, write(), compile(), build(),decision_function(), writes parameters.h
    mapper.py     cost model + policy that picks tiles, width and axis
    tables.py     node and leaf tables, derived from the conifer ensemble
    shard.py      tree and feature assignement on tiles
    precision.py  ap_fixed to integer width and binary point
    checks.py     guards with detailed error messages
    report.py     report reader
    roles.py      per-tile kernel preprocessing directives writer
    devices.py    device records (devices/*.json), independent of toolchain
    platforms.py  locating a platform from the Vitis environment
    tools.py      toolchain discovery, and running one Makefile target
    firmware/     the vendored kernels: common/, axis/, oblique/
    template/     the project Makefile

`writer.py` owns the model class and calls everything else.


## The four stages

| call | needs | gives |
|---|---|---|
| `write()` | nothing | the project, the resolved mapping, and a forward cost estimate |
| `compile()` | `aiecompiler` | the x86 functional correctness |
| `decision_function(X)` | `x86simulator` | scores, bit-accurate to the hardware arithmetic |
| `build(simulate=False)` | `aiecompiler` | the placement, tile memory and program memory of the mapped design |
| `build(X)` | `aiesimulator` | above, plus cycles: `cyc_per_sample`, `latency_ss_ns`, `slowest_tile_ratio`, and `X` scored |

`read_report()` returns whatever stage is on disk.

`build(simulate=False)` stops after the hardware compile: the placement and the memory a tile needs, with no stimulus to supply and no cycle-accurate run to wait for. The simulator is cycle-accurate, so its share grows with `NSamples` while the compile's does not.
`build(X)` (`simulate=True` is the default) compiles and launches the cycle-accurate run, computing scores. Read them back with `read_scores(simulator='aie')`. With no `X` it simulates whatever `data/x.dat` holds, or zeros (the timing does not depend on the data).

## Definitions

- **tile** - one AI Engine core, with its own program and 64 kB of data memory. A VEK280 has 304 of them.
- **PLIO** - the stream ports between the array and the rest of the chip. They are a scarcer resource than the cores (VEK280 has 112 outgoing ones).
- **memtile** - a shared on-chip buffer that several tiles can read from, each at its own offset. It is how different tiles can get different data from one input port.
- **tree-split** - use more tiles by giving each a subset of the trees. Every tile sees every sample and emits a partial score, and the partials are summed off-array. Shortens latency.
- **sample-split** - use more tiles by giving each a subset of the samples, with the whole ensemble on each. Nothing to sum. Raises throughput.

One **invocation** is one call of a tile's kernel, scoring `VectorWidth` samples against its share of the trees.

## Configuration

Most mapping fields may be `'auto'`, in which case the backend chooses one and reports what it chose (`model.resolved_config()`).

`Priority` always holds a value, because the others are chosen against.

`XilinxPart`, `Platform` and `ElfgenJobs` are for the toolchain and to be manually chosen.

| field | default | meaning |
|---|---|---|
| `Priority` | `latency` | `latency` splits trees across tiles, `throughput` splits samples. Impacts the tile count and vector width |
| `NTiles` | `auto` | number of tiles to use |
| `SplitAxis` | `auto` | `tree` or `sample` |
| `VectorWidth` | `auto` | samples per invocation, auto fills a vector register or two depending on `Priority` |
| `PlioRate` | `auto` | offered input rate in MHz, at most half the array clock |
| `TreesPerTile` | `auto` | how many trees each tile takes under tree-split |
| `NSamples` | `auto` | rows the graph is compiled to score in one run |
| `Shard` | `auto` | narrow each tile's feature rows to a window, `auto` or `False` |
| `Feed` | `auto` | `memtile` shares fewer PL inputs across the array, `plio` gives each tile a PL port |
| `XilinxPart` | `xcve2802-vsvh1760-2MP-e-S` | selects the target device part (see AMD's documentation for more details)  |
| `Platform` | unset | overrides the `.xpfm` the toolchain builds against. The device record names one, so this is only for a custom or renamed platform, or an absolute path |
| `ElfgenJobs` | unset | caps `aiecompiler` ELF generation concurrency |
| `Precision` | `ap_fixed<16,5,AP_RND_CONV,AP_SAT>` | the compare path, and the fallback for the three below. Must be `ap_fixed<16,I,AP_RND_CONV,AP_SAT>` |
| `InputPrecision` | `Precision` | the feature rows a tile reads |
| `ThresholdPrecision` | `Precision` | the node thresholds they are compared against |
| `WeightPrecision` | `Precision` | the projection weights, oblique only |
| `ScorePrecision` | `ap_fixed<32,16,AP_RND_CONV,AP_SAT>` | the accumulator, which must hold the sum over all trees. Must be `ap_fixed<32,I,AP_RND_CONV,AP_SAT>` |

`AP_RND_CONV,AP_SAT` is the only supported mode: the tables are quantized round-half-to-even and saturating.

## Kernels

| model | kernel | tiles |
|---|---|---|
| axis-aligned | per-tree table layout, multi-tile | 1–112 |
| oblique, weights in {0, ±1} | partial-projection basis, multi-tile | 1–112 |

Neither kernel walks a tree. Every node in a tile's trees is evaluated, each failing node clears the leaves it rules out from a per-tree bitmask, and the surviving lowest bit is the exit leaf. No data-dependent branches and no pointer chasing, which is what makes the work vectorize across samples.

That exit-leaf scheme is the one **QuickScorer** introduced (Lucchese et al., *QuickScorer: A Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees*, SIGIR 2015). We then built on top of it to vectorize across a sample group, the multi-word bitmask that carries depth, the leaf select chain, the split across tiles, and the oblique extension.

## Limits

Raised at `write()`:

- `max_depth > 6` not supported yet
- oblique projection weights outside {0, ±1}
- more than two classes, the kernels score one value per sample
- `n_tiles` above the device's tile count, or above the outgoing PLIO channels the platform routes: every tile emits its own partial score on its own channel
- unsupported precision width, rounding or overflow mode

## Metrics

`build()` reports:

| | what it counts |
|---|---|
| `cyc_per_sample` | the kernel's own cycles, per score |
| `throughput_ns_per_sample` | the steady-state invocation period, per score |
| `run_ns_per_sample` | `cyc_per_sample` + graph startup and teardown included. **Not a throughput**. |
| `latency_ss_ns` | one group's residence, from the array accepting its first input word to its last score existing |
| `slowest_tile_ratio` | the busiest tile's cycles over the average tile's. 1.0 means the array is balanced |
| `est_cyc_per_sample`, `est_latency_ss_ns` | reported by `write()` from a cost model fitted on VEK280 measurements. It is still an estimate, use `build()` for real numbers. |

`throughput_ns_per_sample` is the period the array takes from one group's last output to the next group's, over the samples scored during an invocation: `W` on a tree-split, `W x n_tiles` on a sample-split, which is what the estimate divides by too. Its excess over `cyc_per_sample` is reported as `io_ns_per_sample`, the time the array spends not computing; on the shipped examples that is a fraction of a nanosecond, because these designs are compute-bound.

`latency_ss_ns` is one group's residence in the array, from its first input word being accepted to its last score existing: under a tree-split every tile holds the same group, so it runs from the earliest tile accepting it to the last partial appearing. 

## Giving each tile less to read

A tree-split divides the *work* but not the *input*: by default every tile reads every feature of every sample, then waits on rows most of its trees never test.

A tile's trees are fixed when the project is written, so the features they test are known too. The backend chooses which trees go on which tile and reorders the feature rows, so each tile reads one contiguous window instead of the whole sample. The code calls this *sharding*, and `write()` says what it achieved:

    sharded: each tile reads 13 of 16 feature rows at worst (46 across the array)

Sample-split and oblique models never shard.

## `n_samples`

The graph is compiled for a fixed number of rows, so `n_samples` is a property of the project, not of the model (see it as a batch size). `decision_function(X)` still accepts any number of rows. A short `X` is padded, and a long one is split across runs. In both cases, the corresponding `len(X)` scores are returned.

## How `auto` chooses

`Priority` fixes the split axis (`latency` splits trees, `throughput` splits samples) and the vector width. `W` is the group that fills a vector register, 512 bits for latency and 1024 for throughput, so a deeper ensemble takes a narrower group due to a wider leaf bitvector.

The tile count is then the one minimising whichever metric the priority chooses, over powers of two. That is currently simpler than it sounds: nothing in the cost model gets worse with more tiles, so it always lands on as many as `auto` allows, capped to keep compile time down, with `NTiles` pinning any other count. We implement it as a search because the terms that would make more tiles worse (e.g., off-array summation, PLIO pressure, memtile bandwidth shared across readers) may be modelled in the future.