import csv
import glob
import json
import logging
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)

_TS = re.compile(r"^T\s+(\S+)\s*(\S*)\s*$")
# aiesimulator mixes units within one file, so the unit column has to be read.
_UNIT_NS = {"": 1.0, "s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0, "ps": 1e-3}
_RUNTIME = {"main", "_main_no_exit_init", "_main_init", "_start"}

_NEXT_STAGE = {
    "write": "run compile() to check the project builds, then build() for the mapped design",
    "compile": "run build(simulate=False) for the mapping and tile memory, or build(X) for those plus cycle counts, latency and X scored",
    "build": None,
}

# A hardware compile leaves the mapping behind without simulating, so the compile stage
# can already carry what the generic hint offers to go and get.
_NEXT_MAPPED = "run build(X) for cycle counts, latency and X scored"


def read_plio(path):
    """Timestamps in ns and integer values from a PLIO .dat"""
    ts, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == "TLAST":
                continue
            m = _TS.match(line)
            if m:
                ts.append(float(m.group(1)) * _UNIT_NS[m.group(2)])
            else:
                vals.extend(int(t) for t in line.split())
    ts = np.asarray(ts)
    if np.any(np.diff(ts) < 0):
        raise RuntimeError(f"{path}: timestamps go backwards, a unit was mishandled")
    return ts, np.asarray(vals, dtype=np.int64)


def _cores(sim_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(sim_dir, "profile_funct_*.xml"))):
        root = ET.parse(p).getroot()
        if int(root.findtext("total_cycle_count") or 0) <= 0:
            continue
        fns = []
        for f in root.iter("function"):
            name = (f.findtext("function_name") or "").strip()
            cyc = int(
                f.findtext("function_and_descendants_time/total_cycle_count") or 0
            )
            if name and name not in _RUNTIME and cyc > 0:
                fns.append((name, int(f.findtext("calls") or 0), cyc))
        if not fns:
            continue
        name, calls, cyc = max(fns, key=lambda f: f[2])
        col, row = os.path.basename(p)[len("profile_funct_") : -len(".xml")].split("_")
        out.append(
            {
                "col": int(col),
                "row": int(row),
                "name": name,
                "calls": calls,
                "cyc": cyc,
                "total": int(root.findtext("total_cycle_count")),
            }
        )
    return out


def _map_sections(path):
    with open(path) as f:
        blocks = [b for b in re.split(r"\n\s*\n", f.read()) if b.strip()]
    out = {}
    for b in blocks:
        rows = list(csv.DictReader(b.strip().splitlines()))
        if rows:
            out[next(iter(rows[0]))] = rows
    return out


def _tile(s):
    m = re.fullmatch(r"\((\d+),\s*(\d+)\)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _sizes(path, key):
    out, core = {}, None
    pat = re.compile(key + r"\s*=\s*(\d+)")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r"Core (\S+)", line)
            if m:
                core = m.group(1)
            m = pat.search(line)
            if m and core:
                out[core] = int(m.group(1))
    return out


def _compile_metrics(work_dir, report):
    map_path = os.path.join(os.path.dirname(work_dir), "Map_Report.csv")
    if os.path.exists(map_path):
        mr = _map_sections(map_path)
        per_tile = {}
        for r in mr.get("BUFFER", []):
            t = _tile(r.get("MEMORY_GROUP", ""))
            if t:
                per_tile[t] = per_tile.get(t, 0) + int(r["SIZE"])
        if per_tile:
            report["tile_memory_bytes"] = dict(sorted(per_tile.items()))
            report["tile_memory_bytes_max"] = max(per_tile.values())
        clusters = mr.get("CLUSTER", [])
        if clusters:
            report["placement"] = [
                (c.get("CLUSTER"), _tile(c.get("TILE", ""))) for c in clusters
            ]
            report["n_tiles"] = len(clusters)

    reports = os.path.join(work_dir, "reports")
    pm = _sizes(os.path.join(reports, "report_pm.txt"), r"PM Size Used")
    if pm:
        report["program_memory_bytes"] = max(pm.values())
    heap = _sizes(
        os.path.join(reports, "report_heap.txt"), r"Heap Size Used \(after alignment\)"
    )
    if heap:
        report["heap_bytes"] = max(heap.values())
    stack = _sizes(os.path.join(reports, "report_stack.txt"), r"Stack Size Used")
    if stack:
        report["stack_bytes"] = max(stack.values())


def _invocation_period_ns(sim_dir, W):
    """One group's last output to the next group's

    Measured from the same output timestamps as latency_ss. Taken from the last word of
    a group rather than the first, so it includes the time the port spends draining the
    group it just finished.
    """
    path = os.path.join(sim_dir, "scores.dat")
    if not (W and os.path.isfile(path)):
        return None
    ts, _ = read_plio(path)
    per = np.diff(ts[W - 1 :: W])
    return per if len(per) else None


def _throughput(sim_dir, meta, n_tiles, split_axis, ghz, report):
    """Steady-state cost of a sample, from the period the array holds

    A tree-split invocation scores W samples on every tile at once. A sample-split one
    scores W on each of the tiles. Which it is, is what the estimate divides by too.
    """
    W = meta.get("config", {}).get("vector_width")
    per = _invocation_period_ns(sim_dir, W)
    if per is None:
        report["throughput_ns_per_sample"] = report.get("run_ns_per_sample")
        report["throughput_note"] = (
            "whole run: one group is too few to time a period, so this carries the "
            "graph startup and falls with n_samples"
        )
        return
    samples_per_invocation = W * (n_tiles if split_axis == "sample" else 1)
    report["invocation_period_ns"] = float(np.median(per))
    report["invocation_period_sd_ns"] = float(np.std(per))
    report["n_invocations"] = len(per) + 1
    report["throughput_ns_per_sample"] = float(np.median(per) / samples_per_invocation)
    if "cyc_per_sample" in report:
        # What a sample costs beyond the kernel's own time: the array's I/O...
        report["io_ns_per_sample"] = float(
            report["throughput_ns_per_sample"] - report["cyc_per_sample"] / ghz
        )


def _build_metrics(sim_dir, meta, report):
    cores = _cores(sim_dir)
    if not cores:
        return
    n_tiles = meta.get("config", {}).get("n_tiles", len(cores))
    split_axis = meta.get("config", {}).get("split_axis", "tree")
    n_samples = meta.get("n_samples", 0)
    ghz = meta.get("clock_ghz", 1.25)

    report["n_active_cores"] = len(cores)
    if len(cores) != n_tiles:
        logger.warning(
            f"{len(cores)} cores active but the model declares {n_tiles} tiles; "
            f"an idle tile would read as a speedup"
        )
    if n_samples:
        cps = np.asarray([c["cyc"] for c in cores]) / n_samples
        report["cyc_per_sample"] = float(cps.max())
        report["cyc_per_sample_avg"] = float(cps.mean())
        # 1.0 is a perfectly balanced array, 1.05 means the busiest tile does 5% more work
        report["slowest_tile_ratio"] = (
            float(cps.max() / cps.mean()) if cps.mean() else None
        )
        total = max(c["total"] for c in cores)
        report["total_cyc"] = total
        # Every cycle of the core's run. Its excess over the kernel's own time is a
        # fixed graph startup and teardown, around 7300 to 9600 cycles for the examples
        report["run_ns_per_sample"] = float(total / ghz / n_samples)
        _throughput(sim_dir, meta, n_tiles, split_axis, ghz, report)

    per_call_ns = np.asarray([c["cyc"] / c["calls"] / ghz for c in cores if c["calls"]])
    if not per_call_ns.size:
        return
    if len(cores) == 1 or split_axis == "sample":
        # A group lives on one tile, so its residence == tile's own invocation
        report["latency_ss_ns"] = float(per_call_ns.mean())
        report["latency_ss_sd_ns"] = float(per_call_ns.std())
        report["latency_ss_drift_ns_per_group"] = 0.0
    else:
        _tree_split_latency(sim_dir, cores, per_call_ns, meta, report)

    tp = os.path.join(os.path.dirname(sim_dir), "throughput_info.json")
    if os.path.exists(tp):
        with open(tp) as f:
            ports = json.load(f).get("plio_throughput", {})
        report["port_throughput_mbps"] = {
            v.get("port name"): float(str(v.get("Throughput", "0")).split()[0])
            for v in ports.values()
        }


MIN_STEADY_GROUPS = 4


def _slope_per_group(values):
    n = len(values)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = float(np.mean(values))
    denom = sum((i - mx) ** 2 for i in range(n))
    if not denom:
        return 0.0
    return sum((i - mx) * (v - my) for i, v in enumerate(values)) / denom


def _summarise_latency(lat, report, offset=0, note=""):
    """Residence as a fit, not a mean

    A pipelined mapping can hold a steady period while residence climbs, so the mean is
    a function of the run length. The intercept - a group entering with no accumulated
    skew - is not, and the slope is reported beside it.
    """
    lat = list(lat)
    if len(lat) < MIN_STEADY_GROUPS:
        report["latency_ss_note"] = (
            f"unmeasured: {len(lat)} steady-state groups, under "
            f"the {MIN_STEADY_GROUPS} required"
        )
        return
    drift = _slope_per_group(lat)
    mid = offset + (len(lat) - 1) / 2.0
    intercept = float(np.mean(lat)) - drift * mid
    resid = [v - (intercept + drift * (offset + i)) for i, v in enumerate(lat)]
    sd = float(np.std(resid))
    report["latency_ss_ns"] = intercept
    report["latency_ss_sd_ns"] = sd
    report["latency_ss_drift_ns_per_group"] = drift
    if sd > 0.10 * intercept:
        note = (note + "; " if note else "") + (
            f"latency_ss scatters {sd:.0f} ns about its fit, "
            f"{100 * sd / intercept:.0f}% of the intercept"
        )
    if note:
        report["latency_ss_note"] = note


def _tree_split_latency(sim_dir, cores, per_call_ns, meta, report):
    """Residence of a group across the array, from the per-tile output timestamps"""
    n = len(cores)
    W = meta.get("config", {}).get("vector_width") or meta.get("n_samples")
    files = [
        os.path.join(sim_dir, "scores.dat" if i == 0 else f"scores.t{i}.dat")
        for i in range(n)
    ]
    if not (W and all(os.path.isfile(f) for f in files)):
        report["latency_ss_ns"] = float(per_call_ns.max())
        report["latency_ss_note"] = "approximated by the slowest tile's invocation"
        return

    order = _tile_order(cores, n)
    if order is None:
        report["latency_ss_ns"] = float(per_call_ns.max())
        report["latency_ss_note"] = "approximated: cores could not be matched to ports"
        return
    per_call_ns = per_call_ns[order]

    lasts = [read_plio(f)[0][W - 1 :: W] for f in files]
    g = min(len(t) for t in lasts)
    if g < 1:
        report["latency_ss_ns"] = float(per_call_ns.max())
        report["latency_ss_note"] = "approximated: too few groups to time"
        return
    out = np.asarray([t[:g] for t in lasts])
    lat = out.max(axis=0) - (out - per_call_ns[:, None]).min(axis=0)
    # Drop the groups that fill the array and the ones that drain it
    skip = min(len(lat) // 8, max(0, (len(lat) - MIN_STEADY_GROUPS) // 2))
    trimmed = lat[skip : len(lat) - skip] if skip else lat
    _summarise_latency(trimmed, report, offset=skip)


def _tile_order(cores, n):
    """Cores come back in placement order the output ports are in tile order"""
    idx = []
    for c in cores:
        m = re.search(r"_tile_(\d+)$", c["name"])
        if not m:
            return None
        idx.append(int(m.group(1)))
    return np.argsort(idx) if sorted(idx) == list(range(n)) else None


def read_aie_report(output_dir) -> dict:
    """Read whatever stage of AI Engine report is present

    The returned 'stage' says what was found, and 'next_step' says what to run for more.
    """
    report = {"stage": "write"}
    meta_path = os.path.join(output_dir, "aie_model.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        report["estimate"] = meta.get("estimate", {})
        report["n_tiles"] = meta.get("config", {}).get("n_tiles")
        report["estimated_tile_memory_bytes"] = meta.get("memory", {}).get("total")

    for d in ("build_hw", "build_x86"):
        work = os.path.join(output_dir, d, "Work")
        if os.path.isdir(work):
            report["stage"] = "compile"
            _compile_metrics(work, report)
            break

    sim_dir = os.path.join(output_dir, "build_hw", "aiesimulator_output")
    if os.path.isdir(sim_dir) and glob.glob(
        os.path.join(sim_dir, "profile_funct_*.xml")
    ):
        report["stage"] = "build"
        _build_metrics(sim_dir, meta, report)

    report["next_step"] = _NEXT_STAGE.get(report["stage"])
    if report["stage"] == "compile" and "tile_memory_bytes_max" in report:
        report["next_step"] = _NEXT_MAPPED
    return report
