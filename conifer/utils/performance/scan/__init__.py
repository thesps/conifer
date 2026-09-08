'''
Performance-scan tooling: generate random conifer models, synthesize them and collect
latency/resource/build measurements to train the conifer.utils.performance estimators.

Opt-in and not imported by conifer.utils.performance; install extras with `pip install conifer[scan]`.
Heavy dependencies (pandas, pyyaml, pyarrow) are imported lazily by the functions that need them.
'''
from conifer.utils.performance.scan.generators import RandomTree, random_model
from conifer.utils.performance.scan.spec import Axis, ScanSpec
from conifer.utils.performance.scan.manifest import Manifest, Point, expand_spec
from conifer.utils.performance.scan.build import make_model, build_point, run_point
from conifer.utils.performance.scan.run import run_manifest
from conifer.utils.performance.scan.gather import gather, status_report
from conifer.utils.performance.scan.plan import DummyCostModel, group_into_batches, plan

__all__ = ['RandomTree', 'random_model', 'Axis', 'ScanSpec', 'Manifest', 'Point',
           'expand_spec', 'make_model', 'build_point', 'run_point', 'run_manifest',
           'gather', 'status_report', 'plan', 'group_into_batches', 'DummyCostModel']
