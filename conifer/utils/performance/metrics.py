'''
Feature (ModelMetrics) and label (SynthesisMetrics) schema for the performance estimators
get_model_metrics()/get_model_measurements() are the main entry points
'''
import dataclasses
import os
from dataclasses import dataclass

import numpy as np

from conifer.model import ModelBase
from conifer.utils.fixed_point import ApType, parse_ap_type

# schema versions: bump when a field is added/removed/renamed on ModelMetrics or SynthesisMetrics
MODEL_METRICS_SCHEMA_VERSION = 1
SYNTHESIS_METRICS_SCHEMA_VERSION = 1


@dataclass
class SummaryStatistics:
  '''mean/std/min/max/sum/quartiles of an array of values'''
  mean: float
  std: float
  min: float
  max: float
  sum: float
  quartile_1: float
  quartile_3: float

  def flatten(self, prefix=''):
    return {f'{prefix}{k}': v for k, v in dataclasses.asdict(self).items()}

def array_summary_statistics(d) -> SummaryStatistics:
  '''Get SummaryStatistics from an array'''
  quartiles = np.quantile(d, [0.25, 0.75])
  return SummaryStatistics(mean=np.mean(d), std=np.std(d), min=np.min(d), max=np.max(d),
                           sum=np.sum(d), quartile_1=quartiles[0], quartile_3=quartiles[1])

def get_sparsity_metrics(model) -> SummaryStatistics:
  sparsity = np.array([1 - (tree.n_nodes() - tree.n_leaves()) / (2 ** model.max_depth - 1) for tree_c in model.trees for tree in tree_c])
  return array_summary_statistics(sparsity)

def get_feature_frequency(model : ModelBase):
  def _get_feature_counts_tree(tree):
    features, counts = np.unique(tree.feature, return_counts=True)
    return features, counts
  counts = np.zeros(model.n_features)
  for trees_c in model.trees:
    for tree in trees_c:
      f, c = _get_feature_counts_tree(tree)
      counts[f[f != -2]] += c[f != -2]
  n_nodes = model.n_nodes() - model.n_leaves()
  return counts / n_nodes

def get_feature_frequency_metrics(model : ModelBase) -> SummaryStatistics:
  return array_summary_statistics(get_feature_frequency(model))

@dataclass
class ObliqueMetrics:
  '''Fraction of oblique split nodes and stats of their term counts and weight magnitudes'''
  fraction: float
  terms_mean: float
  terms_std: float
  terms_max: float
  weight_abs_mean: float
  weight_abs_std: float
  weight_abs_max: float

  def flatten(self):
    return {f'oblique_{k}': v for k, v in dataclasses.asdict(self).items()}

def _iter_split_weights(model : ModelBase):
  '''Yield the weight vector of every split (non-leaf) node in the model.'''
  for trees_c in model.trees:
    for tree in trees_c:
      for i, f in enumerate(tree.feature):
        if f != -2:
          yield np.asarray(tree.weight[i], dtype=float)

def get_oblique_metrics(model : ModelBase) -> ObliqueMetrics:
  n_terms, w_abs = [], []
  for w in _iter_split_weights(model):
    nz = w[w != 0]
    n_terms.append(len(nz))
    is_oblique = len(nz) > 1 or np.any(np.abs(nz) != 1)
    if is_oblique:
      w_abs.extend(np.abs(nz).tolist())
  if len(n_terms) == 0:
    n_terms = [0]
  w_abs = w_abs if len(w_abs) > 0 else [0.0]
  n_terms, w_abs = np.array(n_terms, dtype=float), np.array(w_abs, dtype=float)
  return ObliqueMetrics(fraction=float(np.mean(n_terms > 1)),
                        terms_mean=float(np.mean(n_terms)), terms_std=float(np.std(n_terms)),
                        terms_max=float(np.max(n_terms)), weight_abs_mean=float(np.mean(w_abs)),
                        weight_abs_std=float(np.std(w_abs)), weight_abs_max=float(np.max(w_abs)))

@dataclass
class PrecisionMetrics:
  '''Parsed ap_ type of each precision field a conifer config may define; None if unset/unparseable'''
  input: ApType = None
  threshold: ApType = None
  weight: ApType = None
  score: ApType = None

  def flatten(self):
    out = {}
    for name in ('input', 'threshold', 'weight', 'score'):
      t = getattr(self, name)
      out[f'{name}_precision_width'] = float(t.width) if t else np.nan
      out[f'{name}_precision_int_bits'] = float(t.integer_bits) if t else np.nan
    return out

def get_precision_metrics(model : ModelBase) -> PrecisionMetrics:
  cfg = model.config
  return PrecisionMetrics(**{name: parse_ap_type(getattr(cfg, f'{name}_precision', None))
                             for name in ('input', 'threshold', 'weight', 'score')})

@dataclass
class ModelMetrics:
  '''Every feature the performance-scan and estimators derive from a conifer model alone'''
  max_depth: int
  n_trees: int
  n_features: int
  n_nodes: int
  n_leaves: int
  backend: str
  sparsity: SummaryStatistics
  feature_frequency: SummaryStatistics
  oblique: ObliqueMetrics
  precision: PrecisionMetrics
  schema_version: int = MODEL_METRICS_SCHEMA_VERSION

  def flatten(self):
    return {'max_depth': self.max_depth, 'n_trees': self.n_trees, 'n_features': self.n_features,
           'n_nodes': self.n_nodes, 'n_leaves': self.n_leaves, 'backend': self.backend,
           'schema_version': self.schema_version,
           **self.sparsity.flatten('sparsity_'), **self.feature_frequency.flatten('feature_frequency_'),
           **self.oblique.flatten(), **self.precision.flatten()}

def get_model_metrics(model : ModelBase) -> ModelMetrics:
  '''Entry point: every feature derivable from a conifer model alone, no synthesis needed'''
  return ModelMetrics(max_depth=model.max_depth, n_trees=model.n_trees, n_features=model.n_features,
                      n_nodes=model.n_nodes() - model.n_leaves(), n_leaves=model.n_leaves(),
                      backend=model.config.backend, sparsity=get_sparsity_metrics(model),
                      feature_frequency=get_feature_frequency_metrics(model),
                      oblique=get_oblique_metrics(model), precision=get_precision_metrics(model))

def _dir_size(path):
  '''Total size in bytes of every file under path (0 if path doesn't exist).'''
  total = 0
  for dp, _, files in os.walk(path):
    for f in files:
      fp = os.path.join(dp, f)
      if not os.path.islink(fp):
        total += os.path.getsize(fp)
  return total

def _read_build_log(outdir):
  from conifer.backends.common import read_hls_log
  for name in ('vitis_hls.log', 'build.log', 'vivado_hls.log'):
    log = read_hls_log(os.path.join(outdir, name))
    if log:
      return log
  return {}

@dataclass
class SynthesisMetrics:
  '''Every label the performance-scan measures from a built model's HLS/Vivado reports and logs'''
  outcome: str
  reason: str
  latency: int = None
  interval: int = None
  hls_lut: int = None
  hls_ff: int = None
  hls_dsp: int = None
  lut: int = None
  ff: int = None
  dsp: int = None
  build_time_s: float = None
  build_memory_gb: float = None
  disk_bytes: int = None
  schema_version: int = SYNTHESIS_METRICS_SCHEMA_VERSION

  def flatten(self):
    return dataclasses.asdict(self)

def get_model_measurements(model : ModelBase, do_vsynth : bool = True) -> SynthesisMetrics:
  '''
  Entry point: parse a built model's HLS/Vivado reports and build log into SynthesisMetrics
  Safe against a missing output directory or missing/malformed report files
  '''
  outdir = model.config.output_dir
  if not os.path.isdir(outdir):
    return SynthesisMetrics(outcome='error', reason=f'output directory not found: {outdir}')

  try:
    rep = model.read_report() or {}
  except Exception as e:
    return SynthesisMetrics(outcome='hls_fail', reason=f'could not parse HLS report: {type(e).__name__}: {e}')
  try:
    log = _read_build_log(outdir)
  except Exception as e:
    log = {}

  vs = rep.get('vsynth', {})
  if rep.get('latency') is None:
    outcome, reason = 'hls_fail', 'no HLS csynth report'
  elif do_vsynth and not vs:
    outcome, reason = 'vsynth_fail', 'no vivado synth report'
  else:
    outcome, reason = 'ok', ''

  return SynthesisMetrics(outcome=outcome, reason=reason, latency=rep.get('latency'),
                          interval=rep.get('interval'), hls_lut=rep.get('lut'), hls_ff=rep.get('ff'),
                          hls_dsp=rep.get('dsp'), lut=vs.get('lut'), ff=vs.get('ff'), dsp=vs.get('dsp'),
                          build_time_s=log.get('time_seconds'), build_memory_gb=log.get('memory_GB'),
                          disk_bytes=_dir_size(outdir))
