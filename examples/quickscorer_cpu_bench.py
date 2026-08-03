# Compare the CPU-runnable conifer backends on one trained BDT, for correctness
# and speed:
#   - python : floating point reference, with three traversal algorithms:
#              'treewalk' (root-to-leaf, per tree), 'quickscorer' and its
#              cache-friendly block-wise variant 'blockwise-quickscorer'
#   - cpp    : bit-accurate CPU emulation of the hardware arithmetic, treewalk
#   - cpp_qs : the same fixed-point arithmetic, QuickScorer traversal, with and
#              without block-wise blocking
# QuickScorer: https://doi.org/10.1145/2766462.2767733
# Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html

# To get consistent results, force running on a single CPU corey with: 
# chrt -f 99 taskset -c 1 python examples/quickscorer_cpu_bench.py

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import datetime
import logging
import sys
import time
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

N_ESTIMATORS = 1000
MAX_DEPTH = 6
# the score accumulator holds the sum over all trees: leave it enough integer bits
PRECISION = 'ap_fixed<18,8>'
# tau (trees per block) and delta (documents per block) need tuning per ensemble
# and per machine, cf. Table 4 of the paper. The python backend scores delta
# documents at once with numpy, the cpp one at a time, hence the different deltas.
PY_TAU, PY_DELTA = 128, 4096
CPP_TAU, CPP_DELTA = 128, 16

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=1.0,
                                 max_depth=MAX_DEPTH, random_state=0).fit(X_train, y_train)

stamp = int(datetime.datetime.now().timestamp())


def convert(backend, name, **extra):
    '''Convert and compile clf for one backend, or None if it can't build here.
    The python backend needs no compilation, but implements compile() anyway.'''
    try:
        cfg = backend.auto_config()
        cfg.update({'OutputDir': f'prj_{name}_{stamp}'}, **extra)
        model = conifer.converters.convert_from_sklearn(clf, cfg)
        model.compile()
        return model
    except Exception as e:
        print(f'Skipping {name} ({e})')
        return None


# one python model serves all three algorithms: 'Algorithm' is the default for
# decision_function, and the algorithm= argument overrides it per call
py_model = convert(conifer.backends.python, 'python', Algorithm='blockwise-quickscorer',
                   Tau=PY_TAU, Delta=PY_DELTA)
cpp_model = convert(conifer.backends.cpp, 'cpp', Precision=PRECISION)
qs_model = convert(conifer.backends.cpp_qs, 'cpp_qs', Precision=PRECISION)
bwqs_model = convert(conifer.backends.cpp_qs, 'cpp_bwqs', Precision=PRECISION,
                     Tau=CPP_TAU, Delta=CPP_DELTA)

scorers = {}  # label -> callable(X) -> scores, in report order
if py_model is not None:
    for label, algorithm in [('python/treewalk', 'treewalk'),
                             ('python/quickscorer', 'quickscorer'),
                             ('python/bwqs', 'blockwise-quickscorer')]:
        scorers[label] = lambda X, a=algorithm: py_model.decision_function(X, algorithm=a)
for label, model in [('cpp', cpp_model), ('cpp_qs', qs_model), ('cpp_qs/bwqs', bwqs_model)]:
    if model is not None:
        scorers[label] = model.decision_function
scorers['sklearn'] = clf.decision_function

scores = {label: fn(X_test) for label, fn in scorers.items()}


# --- correctness -------------------------------------------------------------
def summarize(a, b):
    '''Compare two scorers by label, skipping any that didn't build.'''
    if a not in scores or b not in scores:
        return
    ya, yb = scores[a], scores[b]
    if np.array_equal(ya, yb):
        print(f'{a:18s} vs {b:16s}: agree 100% ({len(ya)}/{len(ya)})')
    else:
        abs_diff = np.abs(ya - yb)
        rel_diff = abs_diff / np.where(np.abs(yb) > 0, np.abs(yb), 1)
        print(f'{a:18s} vs {b:16s}: differ. max abs diff: {abs_diff.max():.4g}, '
              f'max rel diff: {rel_diff.max():.4g}')


print(f'\n--- correctness ({N_ESTIMATORS} trees, depth {MAX_DEPTH}, '
      f'precision {PRECISION}) ---')
# the three python traversals sum the same floating point leaf values: bit-identical
summarize('python/quickscorer', 'python/treewalk')
summarize('python/bwqs', 'python/treewalk')
# and they track sklearn up to floating point roundoff
summarize('python/treewalk', 'sklearn')
# the compiled backends share the arithmetic and the summation order: bit-identical
summarize('cpp_qs', 'cpp')
summarize('cpp_qs/bwqs', 'cpp')
# fixed point vs floating point: differ by the quantization. The largest
# differences come from the few samples that sit within one quantization step of
# a split threshold and take the other branch
summarize('cpp', 'python/treewalk')


# --- speed -------------------------------------------------------------------
def time_us(fn, X, repeats=3):
    '''Best-of-N wall time per sample, in microseconds'''
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e6


# the python treewalk loops over every node in python: time it on a subset
SLOW = {'python/treewalk'}
timings = {label: time_us(fn, X_test[:500], repeats=1) if label in SLOW
           else time_us(fn, X_test)
           for label, fn in scorers.items()}

# NB cpp and cpp_qs run the same ap_fixed arithmetic, but the cpp backend also
# supports oblique splits, so it evaluates a full weight . x dot product at every
# node it visits; cpp_qs is axis-aligned only and compares one feature per node.
print(f'\n--- speed ({len(X_test)} samples) ---')
ref = 'python/treewalk' if 'python/treewalk' in timings else next(iter(timings))
for label, t in timings.items():
    print(f'{label:20s} {t:9.2f} us/sample   ({timings[ref] / t:8.1f}x vs {ref})')

if qs_model is not None:
    print(f'\ncpp_qs traversal data structures (Table 1 of the paper): '
          f'{qs_model.nbytes() / 1024**2:.2f} MB')
