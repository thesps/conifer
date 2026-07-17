# Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
# Compare the traversal algorithms of the python backend, and the cpp and
# xilinxhls backends, for correctness and speed:

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer
from conifer.backends.python.quickscorer import QuickScorer
import datetime
import logging
import sys
import time
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

SKIP_BWQS_SWEEP = True

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(
    n_estimators=10000, learning_rate=1.0, max_depth=6, random_state=0
).fit(X_train, y_train)

stamp = int(datetime.datetime.now().timestamp())

# --- python backend, both algorithms ---------------------------------------
py_model = conifer.converters.convert_from_sklearn(
    clf, {"backend": "python", "Algorithm": "quickscorer"}
)
y_qs = py_model.decision_function(X_test)  # quickscorer (from config)
y_tw = py_model.decision_function(X_test, algorithm="treewalk")
y_bw = py_model.decision_function(X_test, algorithm="blockwise-quickscorer")


# --- compiled backends
def try_compiled(backend, name):
    try:
        cfg = backend.auto_config()
        cfg["OutputDir"] = f"prj_{name}_{stamp}"
        model = conifer.converters.convert_from_sklearn(clf, cfg)
        model.compile()
        return model
    except Exception as e:
        print(f"Skipping {name} backend ({e})")
        return None


cpp_model = try_compiled(conifer.backends.cpp, "cpp")
hls_model = None #try_compiled(conifer.backends.xilinxhls, "hls")
y_cpp = cpp_model.decision_function(X_test) if cpp_model is not None else None
# y_hls = hls_model.decision_function(X_test) if hls_model is not None else None

# --- ground truth
y_skl = clf.decision_function(X_test)


# --- correctness -------------------------------------------------------------
def summarize(name, a, b):
    if b is None:
        return
    if np.array_equal(a, b):
        print(f"{name}: agree 100% ({len(a)}/{len(a)})")
    else:
        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / np.where(np.abs(b) > 0, np.abs(b), 1)
        print(
            f"{name}: differ. max abs diff: {abs_diff.max():.4g}, "
            f"max rel diff: {rel_diff.max():.4g}"
        )


print("\n--- correctness ---")
# the python traversals compute the same floating point sums: bit-identical
summarize("quickscorer vs treewalk", y_qs, y_tw)
summarize("bwqs        vs treewalk", y_bw, y_tw)
# both track sklearn up to float roundoff
summarize("quickscorer vs sklearn ", y_qs, y_skl)
# the compiled backends agree with each other bit-for-bit, but differ from the
# floating point scores by the fixed-point quantization
# summarize("cpp         vs hls     ", y_cpp, y_hls)
summarize("quickscorer vs cpp     ", y_qs, y_cpp)


# --- speed -------------------------------------------------------------------
def time_us(fn, X, repeats=3):
    """Best-of-N wall time per sample, in microseconds"""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e6


# the treewalk is slow (pure python loops): time it on a subset
X_small = X_test[:1000]
timings = {}
timings["python/treewalk"] = time_us(
    lambda X: py_model.decision_function(X, algorithm="treewalk"), X_small, repeats=1
)
timings["python/quickscorer"] = time_us(
    lambda X: py_model.decision_function(X, algorithm="quickscorer"), X_test
)
timings["python/bwqs"] = time_us(
    lambda X: py_model.decision_function(X, algorithm="blockwise-quickscorer"), X_test
)
if cpp_model is not None:
    timings["cpp"] = time_us(cpp_model.decision_function, X_test)
if hls_model is not None:
    timings['hls'] = time_us(hls_model.decision_function, X_test)
timings["sklearn"] = time_us(clf.decision_function, X_test)

print(f"\n--- speed ({clf.n_estimators} trees, depth {clf.max_depth}) ---")
base = timings["python/treewalk"]
for name, t in timings.items():
    print(f"{name:20s} {t:10.2f} us/sample   ({base / t:6.1f}x vs treewalk)")

if SKIP_BWQS_SWEEP:
    exit(0)

# --- BWQS: cache-aware blocking ------------
# NOTE: I'm not really sure about the sizes here, this is just for early testing
#       I'll make a more rigorous testing script for once the cpp implementation is done.

L3 = 24 * 1024**2  # 24MB L3 cache
X_big = np.tile(X_test, (4, 1))
qs = QuickScorer(py_model)
n_flat = qs.n_flat
print(f"\n--- BWQS blocking (L3 = {L3 / 1024**2:.0f} MB, {len(X_big)} documents) ---")
print(f"QS data structures (Table 1): {qs.nbytes() / 1024**2:6.2f} MB")
print(
    f"QS result bitvectors v      : {len(X_big) * n_flat * 8 / 1024**2:6.2f} MB "
    f"({'exceeds' if len(X_big) * n_flat * 8 > L3 else 'fits in'} L3)"
)

print(
    f"\n{'tau':>6s} {'delta':>7s} {'v block':>9s} {'structs/blk':>12s} {'us/sample':>10s}"
)
y_ref = None

# tau = trees per block, delta = documents per block. (None, None) is QS.
for tau, delta in [
    (None, None),
    (None, 4096),
    (128, 8192),
    (128, 4096),
    (128, 2048),
    (32, 2048),
    (128, 256),
]:
    scorer = QuickScorer(py_model, tau=tau, delta=delta)
    t = time_us(scorer.decision_function, X_big)
    y_blk = scorer.decision_function(X_big)
    y_ref = y_blk if y_ref is None else y_ref
    assert np.array_equal(y_blk, y_ref), (
        f"blocking changed the scores (tau={tau}, delta={delta})"
    )
    tau_eff = n_flat if tau is None else tau
    delta_eff = len(X_big) if delta is None else delta
    v_block = tau_eff * delta_eff * 8
    s_block = scorer.blocks[0][1].nbytes()
    label = "QS" if tau is None and delta is None else "BWQS"
    print(
        f"{tau_eff:6d} {delta_eff:7d} {v_block / 1024**2:7.2f}MB {s_block / 1024:9.1f}KB {t:10.2f}   ({label})"
    )
print("(all block shapes verified identical bit by bit)")
