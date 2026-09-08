'''
Generate, synthesize and measure a single scan point.
run_point() isolates the build in a child process with a wall-clock timeout and memory cap.
'''
import datetime
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np

from conifer.utils.performance.scan.manifest import point_dir, result_path

_GEN_KWARGS = {'n_trees', 'max_depth', 'n_features', 'n_classes', 'sparsity', 'split',
               'oblique_k', 'oblique_weight_scale', 'threshold_range', 'value_range', 'norm'}
_PROJECT_NAME = 'conifer_prj'

# files worth keeping once a point is measured (everything else is removed by shrink)
DEFAULT_KEEP = {
  f'{_PROJECT_NAME}.json', 'result.json', 'build.log', 'vitis_hls.log', 'vivado_hls.log',
  'vivado.log', 'vivado_build.log', 'vivado_synth.log', 'vivado_synth.rpt', 'util.rpt',
  f'{_PROJECT_NAME}_csynth.rpt', f'{_PROJECT_NAME}_csynth.xml',
}


def _resolve_sparsity(value, rng, n_trees):
  '''Turn a sparsity descriptor ({dist: const|normal, ...}) into a scalar or per-tree array.'''
  if not isinstance(value, dict):
    return value
  dist = value.get('dist', 'const')
  if dist == 'const':
    return float(value.get('value', 0.0))
  if dist == 'normal':
    return np.clip(rng.normal(value.get('mean', 0.5), value.get('std', 0.5), n_trees), 0.0, 0.999)
  raise ValueError(f'unknown sparsity dist {dist!r}')


def _split_params(params):
  '''Separate a point's params into conifer config overrides and random_model kwargs.'''
  from conifer.utils.performance.scan.spec import CONFIG_KEYS
  config = {k: v for k, v in params.items() if k in CONFIG_KEYS}
  gen = {k: v for k, v in params.items() if k in _GEN_KWARGS}
  return config, gen


def make_model(point, root, base_config):
  '''Build the conifer model for a point (no synthesis).'''
  import conifer
  from conifer.utils.performance.scan.generators import random_model
  rng = np.random.default_rng(point.rng_seed())
  config_overrides, gen_kwargs = _split_params(point.params)
  if 'sparsity' in gen_kwargs:
    gen_kwargs['sparsity'] = _resolve_sparsity(gen_kwargs['sparsity'], rng, int(gen_kwargs['n_trees']))
  ed = random_model(rng, **gen_kwargs)
  config = {**base_config, **config_overrides,
            'output_dir': point_dir(root, point.point_id), 'project_name': _PROJECT_NAME}
  return conifer.model.make_model(ed, config)


def shrink(prjdir, keep=None):
  '''Remove everything in the project dir except the whitelisted result/report/log files.'''
  keep = keep or DEFAULT_KEEP
  for dp, _, files in os.walk(prjdir, topdown=False):
    for f in files:
      if f not in keep:
        try:
          os.remove(os.path.join(dp, f))
        except OSError:
          pass
    if dp != prjdir and not os.listdir(dp):
      try:
        os.rmdir(dp)
      except OSError:
        pass


def _identity(point):
  '''Point identity fields, common to every result written for it.'''
  return {'point_id': point.point_id, 'name': point.name, 'trial': point.trial,
          'seed': point.seed, 'params': point.params}


def _provenance(started, finished):
  '''Run provenance fields, common to every result written for a point.'''
  import conifer
  from conifer.backends.common import get_xilinx_version
  return {'conifer_version': str(conifer.__version__), 'xilinx_version': get_xilinx_version(),
          'hostname': socket.gethostname(), 'python': platform.python_version(),
          'started_at': started.isoformat(), 'finished_at': finished.isoformat(),
          'wall_time_s': (finished - started).total_seconds()}


def _write_result(root, point, result):
  '''Write a point's result.json.'''
  with open(result_path(root, point.point_id), 'w') as f:
    json.dump(result, f, indent=2, default=str)
  return result


def build_point(point, root, base_config, do_vsynth=True, keep=None):
  '''Generate, synthesize and measure one point in-process; write and return its result dict.'''
  from conifer.utils.performance import metrics as perf_metrics
  started = datetime.datetime.now()
  odir = point_dir(root, point.point_id)
  os.makedirs(odir, exist_ok=True)
  outcome, reason, measured = 'error', '', {}
  try:
    model = make_model(point, root, base_config)
    # captured even if synthesis below fails, so failed points are still usable for feature-only analysis
    model_metrics = perf_metrics.get_model_metrics(model)
    measured['model_metrics_schema_version'] = model_metrics.schema_version
    measured.update({k: v for k, v in model_metrics.flatten().items() if k != 'schema_version'})

    backend = model.config.backend
    model.write()
    if backend == 'xilinxhls':
      model.build(synth=True, vsynth=do_vsynth)
    else:
      model.build()

    synthesis_metrics = perf_metrics.get_model_measurements(model, do_vsynth)
    outcome, reason = synthesis_metrics.outcome, synthesis_metrics.reason
    measured['synthesis_metrics_schema_version'] = synthesis_metrics.schema_version
    measured.update({k: v for k, v in synthesis_metrics.flatten().items()
                     if k not in ('schema_version', 'outcome', 'reason')})
    shrink(odir, keep)
  except Exception as e:  # a failed point must still leave a result so it is not retried blindly
    outcome, reason = 'error', f'{type(e).__name__}: {e}'
  finished = datetime.datetime.now()
  result = {**_identity(point), **_provenance(started, finished), 'outcome': outcome,
           'reason': reason, **measured}
  return _write_result(root, point, result)


def _write_stub_result(point, root, outcome, reason, started):
  '''Record an outcome for a point whose child process did not produce a result.'''
  os.makedirs(point_dir(root, point.point_id), exist_ok=True)
  finished = datetime.datetime.now()
  result = {**_identity(point), **_provenance(started, finished), 'outcome': outcome, 'reason': reason}
  return _write_result(root, point, result)


def run_point(point, root, base_config, timeout=None, mem_gb=None, do_vsynth=True,
              keep=None, isolate=True):
  '''Run one point, by default in a child process with a timeout and RLIMIT_AS memory cap.'''
  if not isolate:
    return build_point(point, root, base_config, do_vsynth=do_vsynth, keep=keep)

  started = datetime.datetime.now()
  job = {'point': point.to_dict(), 'root': root, 'base_config': base_config,
         'do_vsynth': do_vsynth, 'keep': sorted(keep) if keep else None}
  jobfile = os.path.join(point_dir(root, point.point_id), 'job.json')
  os.makedirs(os.path.dirname(jobfile), exist_ok=True)
  with open(jobfile, 'w') as f:
    json.dump(job, f)

  preexec = None
  if mem_gb:
    import resource
    cap = int(mem_gb * 1024 ** 3)
    preexec = lambda: resource.setrlimit(resource.RLIMIT_AS, (cap, cap))  # noqa: E731
  cmd = [sys.executable, '-m', 'conifer.utils.performance.scan._build_one', jobfile]
  try:
    proc = subprocess.run(cmd, timeout=timeout, preexec_fn=preexec,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except subprocess.TimeoutExpired:
    return _write_stub_result(point, root, 'timeout', f'exceeded {timeout}s', started)
  if os.path.isfile(result_path(root, point.point_id)):
    return json.load(open(result_path(root, point.point_id)))
  # -9/137 = SIGKILL (our RLIMIT_AS cap or the OS OOM killer), -6 = SIGABRT (e.g. malloc failure under memory pressure)
  if proc.returncode in (-9, 137, -6):
    return _write_stub_result(point, root, 'oom', f'child killed (rc={proc.returncode})', started)
  return _write_stub_result(point, root, 'error', f'child exited rc={proc.returncode}', started)
