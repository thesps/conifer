# Compare the CPU-runnable conifer backends on one trained BDT, for correctness and speed:
#   - python : floating point reference, with three traversal algorithms:
#              'treewalk' (root-to-leaf, per tree), 'quickscorer' and its
#              cache-friendly block-wise variant 'blockwise-quickscorer'
#   - cpp    : bit-accurate CPU emulation of the hardware arithmetic, with the
#              same three traversal algorithms, selected the same way
# A final section demonstrates where BWQS actually beats plain QuickScorer, which needs
# an ensemble large enough that its QuickScorer structures outgrow the last level cache.
# QuickScorer: https://doi.org/10.1145/2766462.2767733
# Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html

# To get consistent results, force running on a single CPU corey with: 
# chrt -f 99 taskset -c 1 python examples/quickscorer_cpu_bench.py

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import conifer
import datetime
import logging
import sys
import time
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

N_ESTIMATORS = 10000
MAX_DEPTH = 6
# the score accumulator holds the sum over all trees: leave it enough integer bits
PRECISION = 'ap_fixed<18,8>'
# tau (trees per block) and delta (samples per block) need tuning per ensemble and per
# machine, cf. Table 4 of the paper. delta means different things in the two backends: in
# the python one it is the numpy vectorization width, hence the much larger value
PY_TAU, PY_DELTA = 128, 4096
CPP_TAU, CPP_DELTA = 1024, 16

# Size of a larger ensemble to exhibit BWQS advantage over QS when the QS structures
# exceed cache size. Set to 0 to skip this comparison (adds a couple of minutes)
BWQS_DEMO_TREES = 32768
BWQS_DEMO_SAMPLES = 2000
BWQS_DEMO_TAUS = (256, 1024, 4096)

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=1.0,
                                 max_depth=MAX_DEPTH, random_state=0).fit(X_train, y_train)

stamp = int(datetime.datetime.now().timestamp())


def convert(backend, name, clf, **extra):
    '''Convert and compile a trained model for one backend, or None if it can't build here.
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


py_model = convert(conifer.backends.python, 'python', clf, 
                   Algorithm='blockwise-quickscorer', Tau=PY_TAU, Delta=PY_DELTA)
cpp_model = convert(conifer.backends.cpp, 'cpp', clf, 
                    Precision=PRECISION, Tau=CPP_TAU, Delta=CPP_DELTA)

scorers = {}
for backend, model in [('python', py_model), ('cpp', cpp_model)]:
    if model is None: 
        continue
    for label, algorithm in [(f'{backend}/treewalk', 'treewalk'),
                             (f'{backend}/quickscorer', 'quickscorer'),
                             (f'{backend}/bwqs', 'blockwise-quickscorer')]:
        scorers[label] = lambda X, m=model, a=algorithm: m.decision_function(X, algorithm=a)
scorers['sklearn'] = clf.decision_function

scores = {label: fn(X_test) for label, fn in scorers.items()}


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
# the cpp traversals share the arithmetic and the summation order: bit-identical
summarize('cpp/quickscorer', 'cpp/treewalk')
summarize('cpp/bwqs', 'cpp/treewalk')
# fixed point vs floating point: differ by the quantization
summarize('cpp/treewalk', 'python/treewalk')


def time_us(fn, X, repeats=3):
    '''Best-of-N wall time per sample, in us'''
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e6


# the python treewalk loops over every node in python, time it on a subset
SLOW = {'python/treewalk'}
timings = {label: time_us(fn, X_test[:500], repeats=1) if label in SLOW
           else time_us(fn, X_test)
           for label, fn in scorers.items()}

print(f'\n--- speed ({len(X_test)} samples) ---')
ref = 'python/treewalk' if 'python/treewalk' in timings else next(iter(timings))
for label, t in timings.items():
    print(f'{label:20s} {t:9.2f} us/sample   ({timings[ref] / t:8.1f}x vs {ref})')

if cpp_model is not None:
    print('\ncpp QuickScorer traversal data structures (Table 1 of the paper):')
    for label, algorithm in [('plain QS', 'quickscorer'), ('BWQS', 'blockwise-quickscorer')]:
        print(f'  {label:9s} {cpp_model.nbytes(algorithm) / 1024**2:.2f} MB')


# Plain QuickScorer keeps one set of structures for the whole ensemble and streams all of
# it for every sample, so it slows down substantially once those structures no longer fit
# in the last level cache, BWQS holds one (tau trees) block at a time. Compare the size
# printed below against your machine's cache with (on Linux): lscpu | grep -i 'L3 cache'
if BWQS_DEMO_TREES:
    print(f'\n--- BWQS vs QuickScorer ({BWQS_DEMO_TREES} random forest trees, depth {MAX_DEPTH}) ---')
    print(f'training {BWQS_DEMO_TREES} trees...', flush=True)
    big = RandomForestClassifier(n_estimators=BWQS_DEMO_TREES, 
                                 max_depth=MAX_DEPTH, n_jobs=-1, random_state=0).fit(X_train, y_train)
    big_model = convert(conifer.backends.cpp, 'cpp_bwqs_demo', big, Precision=PRECISION)
    if big_model is None:
        print('cpp backend unavailable, skipping')
    else:
        Xd = np.resize(X_test, (BWQS_DEMO_SAMPLES, X_test.shape[1]))
        mb = big_model.nbytes('quickscorer') / 1024**2
        leaves = [t.n_leaves() for tc in big_model.trees for t in tc]
        qs = time_us(lambda X: big_model.decision_function(X, algorithm='quickscorer'), Xd)
        gold = big_model.decision_function(Xd, algorithm='quickscorer')
        print(f'\nleaves per tree: mean {np.mean(leaves):.1f}, max {max(leaves)} (depth {MAX_DEPTH} allows {2 ** MAX_DEPTH})')
        print(f'QuickScorer structures: {mb:.2f} MB')
        print(f'{"":18s} {"us/sample":>10s} {"vs plain QS":>12s}  bit-exact')
        print(f'{"plain quickscorer":18s} {qs:10.2f} {"":>12s}')
        for tau in BWQS_DEMO_TAUS:
            big_model.config.tau, big_model.config.delta = tau, CPP_DELTA
            t = time_us(lambda X: big_model.decision_function(X, algorithm='blockwise-quickscorer'), Xd)
            ok = np.array_equal(gold, big_model.decision_function(Xd, algorithm='blockwise-quickscorer'))
            print(f'{"bwqs tau=" + str(tau):18s} {t:10.2f} {qs / t:11.2f}x  {ok}')
