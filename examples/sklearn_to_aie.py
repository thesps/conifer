'''
Convert a scikit-learn BDT to an AI Engine project.

write() needs no toolchain. compile(), decision_function() and build() need the Vitis
AI Engine tools on PATH.

Run the toolchain stages with:  python sklearn_to_aie.py --build
'''

import logging
import sys

import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

import conifer

# show backend's INFO lines
logging.getLogger().setLevel(logging.INFO)

X, y = make_hastie_10_2(n_samples=10000, random_state=0)
X_train, y_train, X_test = X[:9000], y[:9000], X[9000:]

clf = GradientBoostingClassifier(n_estimators=32, max_depth=4, random_state=0)
clf.fit(X_train, y_train)

# 'auto' asks the backend to choose the mapping, set any handle explicitly to pin it
cfg = conifer.backends.aie.auto_config()
cfg['OutputDir'] = 'prj_aie'
cfg['Priority'] = 'latency'


model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()

print('\nResolved mapping:')
for key, value in model.resolved_config().items():
    if key in ('priority', 'n_tiles', 'split_axis', 'vector_width', 'trees_per_tile', 'n_samples',
               'shard', 'feed', 'plio_rate'):
        print(f'  {key:14s} {value}')

report = model.read_report()
print(f"\nStage: {report['stage']}")
est = report['estimate']
print(f"  estimated {est['est_cyc_per_sample']:.1f} cyc/sample, "
      f"{est['est_latency_ss_ns']:.0f} ns latency_ss, "
      f"{est['est_throughput_ns_per_sample']:.2f} ns/sample "
      f"({1e3 / est['est_throughput_ns_per_sample']:.1f} M samples/s)")
print(f"  next: {report['next_step']}")

if '--build' not in sys.argv:
    print('\nPass --build to run the toolchain stages.')
    sys.exit()

assert model.compile(), 'aiecompiler failed, see the log named above'

y_aie = model.decision_function(X_test[:256])
y_skl = clf.decision_function(X_test[:256])
same = int(np.sum(np.sign(y_aie.ravel()) == np.sign(y_skl)))
print(f'\nScored {len(y_aie)} samples, {same} of {len(y_aie)} with the same prediction '
      f'as scikit-learn')

assert model.build(X_test[:256]), 'aiesimulator failed, see the log named above'

y_hw = np.asarray(model.read_scores(simulator='aie')).ravel()[:256]
if np.array_equal(y_hw, y_aie):
    print(f'\naiesimulator and x86simulator agree exactly ({len(y_hw)}/{len(y_hw)})')
else:
    same = int(np.sum(y_hw == y_aie))
    print(f'\naiesimulator and x86simulator differ: {same} of {len(y_hw)} equal, '
          f'largest difference {np.max(np.abs(y_hw - y_aie)):.6g}')

report = model.read_report()
print(f"\nStage: {report['stage']}")
print(f"  {report['cyc_per_sample']:.2f} cyc/sample on {report['n_active_cores']} tile(s)")
print(f"  {report['throughput_ns_per_sample']:.2f} ns/sample "
      f"({1e3 / report['throughput_ns_per_sample']:.1f} M samples/s)")
print(f"  latency_ss {report['latency_ss_ns']:.1f} ns "
      f"(simulation drift {report['latency_ss_drift_ns_per_group']:.2f} ns/group)")
print(f"  slowest tile {report['slowest_tile_ratio']:.4f}x the average")
print(f"  tile memory {report['tile_memory_bytes_max']} B of 65536")
