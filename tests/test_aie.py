"""AI Engine backend tests

The equality checks live in test_backends.py with the other backends.
"""

import os
import shutil

import numpy as np
import pytest

import conifer
from conifer.backends.aie import mapper
from conifer.backends.aie import report as rpt

COMPARE = "ap_fixed<16,5,AP_RND_CONV,AP_SAT>"
SCORE = "ap_fixed<32,16,AP_RND_CONV,AP_SAT>"
PRECISIONS = {
    "Precision": COMPARE,
    "InputPrecision": COMPARE,
    "ThresholdPrecision": COMPARE,
    "WeightPrecision": COMPARE,
    "ScorePrecision": SCORE,
}


@pytest.fixture(scope="module")
def skl_model():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_classification(
        n_samples=600, n_features=16, n_informative=10, random_state=0
    )
    clf = GradientBoostingClassifier(n_estimators=32, max_depth=4, random_state=0).fit(
        X[:500], y[:500]
    )
    return clf, X[500:]


def _config(tmp_path, **kwargs):
    cfg = conifer.backends.aie.auto_config()
    cfg["OutputDir"] = str(tmp_path)
    cfg.update(kwargs)
    return cfg


def _replay(model, X):
    """What the emitted tables score, in float"""
    return model.score_p.dequantize(
        model.tables.replay(
            X,
            model.threshold_p,
            model.score_p,
            model.init_predict[0],
            norm=model.norm,
            split_le=(model.splitting_convention == "<="),
        )
    )


def _cpp_reference(model, tmp_path, precisions):
    """The same ensemble on conifer's bit-exact cpp backend"""
    d = {k: getattr(model, k) for k in conifer.model.ModelBase._ensemble_fields}
    d["trees"] = [
        [
            {k: getattr(t, k) for k in conifer.model.DecisionTreeBase._tree_fields}
            for t in tc
        ]
        for tc in model.trees
    ]
    cfg = {"Backend": "cpp", "ProjectName": "golden", "OutputDir": str(tmp_path)}
    cfg.update(precisions)
    ref = conifer.model.make_model(d, cfg)
    ref.compile()
    return ref


### project


def test_config_roundtrip(skl_model, tmp_path):
    """Passing resolved_config() back must give an identical project."""
    clf, _ = skl_model
    a = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path / "a", NTiles=4, SplitAxis="tree")
    )
    a.write()

    # Pass the resolved dict back whole. Merging its keys into a fresh auto_config()
    # would prove nothing: that dict still carries the CamelCase alternates set to
    # 'auto', those win, and both sides then resolve to the same mapping by coincidence.
    back = a.resolved_config()
    assert "auto" not in [str(v) for v in back.values()]
    back["output_dir"] = str(tmp_path / "b")
    b = conifer.converters.convert_from_sklearn(clf, back)
    b.write()
    assert (b.n_tiles, b.W, b.split_axis) == (a.n_tiles, a.W, a.split_axis)
    for f in ("src/parameters.h", "aie_model.json"):
        pa = (tmp_path / "a" / f).read_text().replace(str(tmp_path / "a"), "")
        pb = (tmp_path / "b" / f).read_text().replace(str(tmp_path / "b"), "")
        assert pa == pb, f"{f} differs between auto and resolved"


### tables


@pytest.mark.parametrize("max_depth,n_features", [(4, 16), (1, 10), (3, 128)])
def test_tables_match_python_backend(tmp_path, max_depth, n_features):
    """The emitted tables against conifer's own python backend, up to quantization.
    """
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_classification(
        n_samples=600,
        n_features=n_features,
        n_informative=n_features // 2,
        random_state=0,
    )
    clf = GradientBoostingClassifier(
        n_estimators=16, max_depth=max_depth, random_state=0
    ).fit(X[:500], y[:500])
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=4, SplitAxis="tree")
    )
    model.write()

    ref = np.asarray(
        conifer.converters.convert_from_sklearn(clf).decision_function(X[500:])
    ).ravel()
    got = _replay(model, X[500:])
    assert np.all(np.sign(ref) == np.sign(got))
    assert np.max(np.abs(ref - got)) < 0.01


@pytest.mark.parametrize("n_tiles", [2, 16])
def test_shard_preserves_the_score(skl_model, tmp_path, n_tiles):
    """Sharding permutes trees and feature rows so a tile reads a window rather than
    every row. The partials it produces must still sum to the unsharded score.
    """
    from conifer.backends.aie.shard import Sharding

    clf, _ = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf, _config(tmp_path, NTiles=n_tiles, SplitAxis="tree")
    )
    X = np.random.default_rng(0).uniform(-8, 8, size=(48, model.n_features))
    mismatches = model.sharding.verify(
        X,
        model.threshold_p,
        model.score_p,
        model.init_predict[0],
        norm=model.norm,
        split_le=(model.splitting_convention == "<="),
    )
    assert len(mismatches) == 0

    identity = Sharding(
        model.tables,
        model.n_trees_padded,
        model.n_features_padded,
        n_tiles,
        fperm=list(range(model.n_features_padded)),
        optimize=False,
    )
    assert model.sharding.total_rows < identity.total_rows


### mapping


@pytest.mark.parametrize(
    "max_depth,feat_bytes,latency,throughput",
    [
        (4, 2, 32, 64),  # a 16-bit bitvector against a 16-bit feature word
        (5, 2, 16, 32),  # the bitvector widens and the group halves
        (4, 4, 16, 32),  # a 32-bit feature word binds where the bitvector does not
    ],
)
def test_vector_width(max_depth, feat_bytes, latency, throughput):
    """W is not searched: it is the group whose inner-loop vector fills 512 bits for
    latency and 1024 for throughput. A lane carries the result bitvector or the feature
    word it tests, whichever is wider.
    """
    for priority, expect in (("latency", latency), ("throughput", throughput)):
        assert mapper.vector_width(priority, max_depth, feat_bytes) == expect
        _, W, _ = mapper.choose_mapping(
            32, max_depth, 16, feat_bytes, 4, priority, max_tiles=304, clock_ghz=1.25
        )
        assert W == expect, "cost model shouldn't influence W"


def test_estimate(skl_model, tmp_path):
    """Test fitted cost law on two real VEK280 measurements."""
    for depth, trees_per_tile, W, measured in ((4, 32, 32, 5354.0), (6, 1, 8, 891.4)):
        got = mapper.invocation_cycles(16, depth, trees_per_tile, W, 2)
        assert abs(got - measured) / measured < 0.10, (
            f"depth {depth}, W {W}: {got} against {measured}"
        )

    # + sanity check that each priority is best wrt its own metric
    clf, _ = skl_model
    est = {
        p: conifer.converters.convert_from_sklearn(
            clf, _config(tmp_path / p, Priority=p)
        ).estimate
        for p in ("latency", "throughput")
    }
    assert est["latency"]["est_latency_ss_ns"] < est["throughput"]["est_latency_ss_ns"]
    assert (
        est["throughput"]["est_throughput_ns_per_sample"]
        < est["latency"]["est_throughput_ns_per_sample"]
    )


@pytest.mark.parametrize("asked", [1, 333, None])
def test_batch_rounding(skl_model, tmp_path, asked):
    """n_samples is a batch, and a run is W samples on one tile under tree-split or on
    every tile under sample-split. Round up to a whole number of runs, do not refuse.
    """
    from conifer.backends.aie.writer import DEFAULT_BATCH

    clf, _ = skl_model
    kw = {} if asked is None else {"NSamples": asked}
    model = conifer.converters.convert_from_sklearn(clf, _config(tmp_path, **kw))
    step = model.W * (model.n_tiles if model.split_axis == "sample" else 1)
    assert model.batch_step == step
    assert model.n_samples % step == 0
    if asked is None:
        assert min(DEFAULT_BATCH, step) <= model.n_samples < DEFAULT_BATCH + step
    else:
        assert asked <= model.n_samples < asked + step

    model.write()
    with pytest.raises(ValueError, match="exceeds"):
        model.write_input(np.zeros((model.n_samples + 1, model.n_features)))


### stages and the report


def test_decision_function(skl_model, tmp_path):
    """One score per row whatever the batch: a long X is split across runs and a short
    one padded. The row counts here do not divide a batch.
    """
    clf, X = skl_model
    model = conifer.converters.convert_from_sklearn(
        clf,
        _config(
            tmp_path / "aie", NSamples=32, NTiles=4, SplitAxis="tree", **PRECISIONS
        ),
    )
    assert model.compile()
    ref = _cpp_reference(model, tmp_path / "cpp", PRECISIONS)

    for n_rows in (1, 33, 70):
        y = model.decision_function(X[:n_rows])
        assert len(y) == n_rows, f"asked for {n_rows} scores, got {len(y)}"
        y_ref = np.asarray(ref.decision_function(X[:n_rows])).ravel()
        np.testing.assert_allclose(y, y_ref, atol=0, rtol=0)

    # asking for a run that did not happen must not fall back to the one that did
    with pytest.raises(FileNotFoundError):
        model.read_scores(simulator="aie")

def test_latency_ss_is_a_fit_not_a_mean():
    """A pipelined mapping can hold a steady period while residence climbs, so the mean
    is a function of the run length and the intercept is not. A trimmed window must
    still report the intercept rather than its own accumulated skew.
    """

    def drifting(n, base=300.0, drift=6.0):
        return [base + drift * i for i in range(n)]

    fits = []
    for n in (8, 16, 32, 64):
        r = {}
        rpt._summarise_latency(drifting(n), r)
        fits.append(r["latency_ss_ns"])
    assert max(fits) - min(fits) < 1e-6, f"intercept moved with run length: {fits}"

    r = {}
    rpt._summarise_latency(drifting(32, drift=6.4), r)
    assert r["latency_ss_ns"] == pytest.approx(300.0)
    assert r["latency_ss_drift_ns_per_group"] == pytest.approx(6.4)

    full, trimmed = {}, {}
    rpt._summarise_latency(drifting(32), full)
    rpt._summarise_latency(drifting(32)[4:28], trimmed, offset=4)
    assert trimmed["latency_ss_ns"] == pytest.approx(full["latency_ss_ns"])

    too_few = {}
    rpt._summarise_latency([300.0, 306.0], too_few)
    assert "latency_ss_ns" not in too_few and "unmeasured" in too_few["latency_ss_note"]


### the Vitis environment


def test_find_platform(tmp_path, monkeypatch):
    """settings64.sh sets XILINX_VITIS but not PLATFORM_REPO_PATHS"""
    from conifer.backends.aie import platforms

    for var in ("PLATFORM_REPO_PATHS", "XILINX_VITIS", "XILINX_HLS"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="settings64|PLATFORM_REPO_PATHS"):
        platforms.resolve_platform("vek280_base")

    base = tmp_path / "Vitis" / "base_platforms"
    for name in ("vek280_base", "xilinx_vek280_base_202610_1"):
        os.makedirs(base / name)
        open(base / name / f"{name}.xpfm", "w").close()
    monkeypatch.setenv("XILINX_VITIS", str(tmp_path / "Vitis"))
    # prefer over versioned directory
    assert platforms.find_platform("vek280_base").endswith(
        "vek280_base/vek280_base.xpfm"
    )

    shutil.rmtree(base / "vek280_base")
    assert platforms.find_platform("vek280_base") == str(
        base / "xilinx_vek280_base_202610_1" / "xilinx_vek280_base_202610_1.xpfm"
    )
