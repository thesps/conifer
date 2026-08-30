'''
Convert an oblique ydf BDT to an AI Engine project.

write() needs no toolchain. compile(), decision_function() and build() need the Vitis
AI Engine tools on PATH.

Run the toolchain stages with:  python ydf_to_aie.py --build
'''

import logging
import sys

import numpy as np
import ydf
from sklearn.datasets import make_hastie_10_2

import conifer

# show backend's INFO lines
logging.getLogger().setLevel(logging.INFO)

X, y = make_hastie_10_2(n_samples=10000, random_state=0)
y = y == 1  # ydf wants boolean labels, Hastie gives {-1, 1}

model = ydf.GradientBoostedTreesLearner(
    num_trees=100,
    max_depth=3,
    split_axis='SPARSE_OBLIQUE',
    sparse_oblique_weights='BINARY',
    apply_link_function=False,
    label='y',
).train({'x': X[:9000], 'y': y[:9000]})
X_test = X[9000:]

COMPARE = 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>'
precision = {'Precision': COMPARE,
             'InputPrecision': COMPARE,
             'ThresholdPrecision': COMPARE,
             'WeightPrecision': 'ap_fixed<16,3,AP_RND_CONV,AP_SAT>',
             'ScorePrecision': 'ap_fixed<32,16,AP_RND_CONV,AP_SAT>'}

cfg = conifer.backends.aie.auto_config()
cfg.update(precision)
cfg['OutputDir'] = 'prj_ydf_aie'
cfg['Priority'] = 'latency'
# to save time
cfg['NSamples'] = 128

aie_model = conifer.converters.convert_from_ydf(model, cfg)
aie_model.write()

resolved = aie_model.resolved_config()
print(f'\nFamily: {aie_model.family}')
for key in ('priority', 'n_tiles', 'split_axis', 'vector_width', 'trees_per_tile',
            'n_samples', 'shard', 'feed'):
    print(f'  {key:14s} {resolved[key]}')

report = aie_model.read_report()
est = report['estimate']
print(f"\nStage: {report['stage']}")
print(f"  estimated {est['est_cyc_per_sample']:.1f} cyc/sample, "
      f"{est['est_latency_ss_ns']:.0f} ns latency_ss, "
      f"{est['est_throughput_ns_per_sample']:.2f} ns/sample "
      f"({1e3 / est['est_throughput_ns_per_sample']:.1f} M samples/s)")

print(f"  {aie_model.basis['basis_n']} basis entries, built on every tile")

if '--build' not in sys.argv:
    print('\nPass --build to run the toolchain stages.')
    sys.exit()

assert aie_model.compile(), 'aiecompiler failed, see the log named above'

# conifer's cpp backend at the same precision is the reference: it emulates the same
# fixed-point arithmetic on the CPU, so the two should agree bit for bit.
cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg.update(precision)
cpp_cfg['OutputDir'] = 'prj_ydf_cpp'
cpp_model = conifer.converters.convert_from_ydf(model, cpp_cfg)
cpp_model.compile()

y_aie = np.asarray(aie_model.decision_function(X_test[:256])).ravel()
y_cpp = np.asarray(cpp_model.decision_function(X_test[:256])).ravel()

if np.array_equal(y_aie, y_cpp):
    print(f'\nAIE x86sim and cpp agree bit for bit ({len(y_aie)}/{len(y_aie)})')
else:
    n = int(np.sum(y_aie == y_cpp))
    print(f'\nAIE x86sim and cpp differ: {n} of {len(y_aie)} equal, '
          f'largest difference {np.max(np.abs(y_aie - y_cpp)):.6g}')

n_hw = min(256, aie_model.n_samples)
assert aie_model.build(X_test[:n_hw]), 'aiesimulator failed, see the log named above'

y_hw = np.asarray(aie_model.read_scores(simulator='aie')).ravel()[:n_hw]
if np.array_equal(y_hw, y_cpp[:n_hw]):
    print(f'\naiesimulator agrees with cpp bit for bit ({n_hw}/{n_hw})')
else:
    same = int(np.sum(y_hw == y_cpp[:n_hw]))
    print(f'\naiesimulator and cpp differ: {same} of {n_hw} equal, '
          f'largest difference {np.max(np.abs(y_hw - y_cpp[:n_hw])):.6g}')

report = aie_model.read_report()
print(f"\nStage: {report['stage']}")
print(f"  {report['cyc_per_sample']:.2f} cyc/sample on {report['n_active_cores']} tile(s)")
print(f"  {report['throughput_ns_per_sample']:.2f} ns/sample "
      f"({1e3 / report['throughput_ns_per_sample']:.1f} M samples/s)")
print(f"  latency_ss {report['latency_ss_ns']:.1f} ns "
      f"(simulation drift {report['latency_ss_drift_ns_per_group']:.2f} ns/group)")
print(f"  slowest tile {report['slowest_tile_ratio']:.4f}x the average")
print(f"  tile memory {report['tile_memory_bytes_max']} B of 65536")
