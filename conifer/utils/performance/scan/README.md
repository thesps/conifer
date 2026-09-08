# conifer.utils.performance.scan

Tooling to (re)generate the training data for the `conifer.utils.performance` estimators:
synthesize many randomised conifer models and record their latency, resources and build cost.

This subpackage is **opt-in** and is not imported by `conifer` or `conifer.utils.performance`.
Install the extras first:

```bash
pip install conifer[scan]
```

## Concepts

| Object | What it is |
|---|---|
| **ScanSpec** (`spec.py`) | Declarative, serialisable (`.yml`/`.json`) description of the scan: axes + sampling mode + trials + base conifer config. |
| **Point** (`manifest.py`) | One model to generate and synthesize, with a stable `point_id` (hash of its parameters). |
| **Manifest** (`manifest.jsonl`) | The explicit, ordered list of points a spec expands to. Everything downstream is keyed on `point_id`. |
| **result.json** | One per point: outcome, targets (latency/LUT/FF/...), model metrics and provenance. Its `schema_version` is a hash of its own field names (`schema.py`), so it changes automatically whenever the labels or features change. |

A scan directory holds `spec.yml`, `manifest.jsonl`, `points/<point_id>/` and, after `gather`, `results.csv`.

## Local usage

```bash
# 1. expand a spec into an explicit manifest
python -m conifer.utils.performance.scan expand \
    examples/performance_estimates/scans/local_smoke.yml  /data/scan1

# 2. (optional) predicted cost of the whole scan
python -m conifer.utils.performance.scan plan /data/scan1

# 3. run it (resumable; re-running skips points that already have a result.json)
source /opt/Xilinx/Vivado/2024.1/settings64.sh
python -m conifer.utils.performance.scan run /data/scan1 -j 8 --timeout 3600 --mem-gb 32

# 4. progress / failures, then aggregate
python -m conifer.utils.performance.scan status /data/scan1
python -m conifer.utils.performance.scan gather /data/scan1     # -> results.csv (+ .parquet)
```

Each point is built in a child process with a wall-clock `--timeout` and an `RLIMIT_AS`
`--mem-gb` cap that a runaway HLS/Vivado subprocess inherits; a point that times out, is
killed or errors still writes a `result.json` recording the outcome, so `--resume` (default)
never retries it blindly. Pass `--no-isolate` to build in-process for debugging.

## Running at scale (batched, one runner per node)

The manifest is the queue. To spread a large scan over N nodes:

```bash
# balance points across shards by predicted cost (longest-processing-time first)
python -m conifer.utils.performance.scan plan /data/scan1 --hours 2      # writes shards/shard_NN.jsonl

# on node i (identical command everywhere, only the shard file differs):
python -m conifer.utils.performance.scan run /data/scan1 --shard-file shards/shard_0i.jsonl -j <cores>
```

or, without a cost model, a stateless round-robin split:

```bash
python -m conifer.utils.performance.scan run /data/scan1 --shard 3/16 -j <cores>
```

Notes for a scheduler (Kubernetes Jobs, Slurm array, ...):

- Submit shards as an indexed/array job with `parallelism` capped by the **HLS licence
  concurrency**, which is usually the real limit rather than CPU count.
- Size shards so a node runs ~1-3 h: long enough to amortise image pull and licence
  checkout, short enough to bound the loss from a pre-emption.
- Point outputs go to shared storage (`/data/scan1` on a networked filesystem or synced
  after each run). `--resume` then makes a re-run of any shard, anywhere, cheap.
- Within a node, `-j` pulls points dynamically from the shard, so a node with fast points
  simply does more — there is no "wait for the slowest point in the batch" stall.
- `plan`/`group_into_batches` currently balance shards on a constant placeholder
  (`plan.DummyCostModel`) — pass a real `cost_model` (predicting from a point's params) to
  `plan()`/`group_into_batches()` once build time/memory/disk estimators exist, for tighter
  balance than an even split.
