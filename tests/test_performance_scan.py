'''Tests for conifer.utils.performance.scan that do not require an HLS toolchain.'''
import json

import numpy as np
import pytest

import conifer
from conifer.utils.performance.scan import (Axis, DummyCostModel, Manifest, ScanSpec,
                                            group_into_batches, make_model, random_model)
from conifer.utils.performance.scan.generators import RandomTree
from conifer.utils.performance import metrics as perf_metrics

CFG = {'backend': 'cpp', 'output_dir': '/tmp/cnf_scan_test', 'project_name': 'p',
       'precision': 'ap_fixed<18,8>'}


def _xilinxhls_cfg(output_dir, project_name='conifer_prj'):
  return {'backend': 'xilinxhls', 'output_dir': str(output_dir), 'project_name': project_name,
          'precision': 'ap_fixed<18,8>', 'xilinx_part': 'xcvu9p-flgb2104-2L-e',
          'clock_period': 5, 'unroll': True}


# ---- generators ----------------------------------------------------------
@pytest.mark.parametrize('split', ['axis', 'oblique'])
def test_random_model_loads(split):
  rng = np.random.default_rng(0)
  ed = random_model(rng, n_trees=4, max_depth=4, n_features=8, sparsity=0.3, split=split)
  model = conifer.model.make_model(ed, CFG)
  assert model.n_trees == 4 and model.max_depth == 4
  assert model.is_oblique() == (split == 'oblique')
  for tc in model.trees:
    for t in tc:
      n = len(t.feature)
      assert all(len(x) == n for x in (t.weight, t.threshold, t.value, t.children_left, t.children_right))
      for i, cl in enumerate(t.children_left):
        assert (t.feature[i] == -2) == (cl == -1)


def test_sparsity_prunes_nodes():
  rng = np.random.default_rng(1)
  full = RandomTree(rng, max_depth=5, n_features=4, sparsity=0.0).to_dict()
  sparse = RandomTree(rng, max_depth=5, n_features=4, sparsity=0.6).to_dict()
  assert len(sparse['feature']) < len(full['feature'])


def test_generation_is_deterministic():
  a = random_model(np.random.default_rng(42), n_trees=3, max_depth=4, n_features=5, split='oblique')
  b = random_model(np.random.default_rng(42), n_trees=3, max_depth=4, n_features=5, split='oblique')
  assert a == b


# ---- spec / manifest ---------------------------------------------------
def _spec():
  return ScanSpec(name='t', mode='grid', n_trials=2, seed=7,
                  generator_defaults={'n_features': 6, 'split': 'axis'},
                  axes=[Axis('n_trees', 'grid', values=[5, 10]),
                        Axis('max_depth', 'grid', values=[2, 3])])


def test_spec_roundtrip_yaml_json(tmp_path):
  s = _spec()
  for path in (tmp_path / 's.json', tmp_path / 's.yml'):
    s.to_yaml(path) if str(path).endswith('yml') else s.to_json(path)
    assert ScanSpec.load(str(path)).to_dict() == s.to_dict()


def test_spec_load_rejects_unknown_extension(tmp_path):
  path = tmp_path / 's.txt'
  path.write_text('{}')
  with pytest.raises(ValueError):
    ScanSpec.load(str(path))


def test_grid_expansion_count_and_stable_ids():
  pts = _spec().expand()
  assert len(pts) == 2 * 2 * 2  # n_trees x max_depth x n_trials
  assert [p.point_id for p in pts] == [p.point_id for p in _spec().expand()]


def test_random_expansion_deterministic():
  s = ScanSpec(name='r', mode='random', n_samples=5, seed=3,
               axes=[Axis('n_trees', 'randint', low=5, high=50),
                     Axis('max_depth', 'randint', low=2, high=6)])
  assert [p.params for p in s.expand()] == [p.params for p in s.expand()]


def test_manifest_write_load_shard_pending(tmp_path):
  manifest = Manifest.from_spec(_spec())
  manifest.write(str(tmp_path))
  assert (tmp_path / 'spec.yml').is_file() and not (tmp_path / 'spec.json').exists()
  m = Manifest.load(str(tmp_path))
  assert len(m) == 8
  shards = [m.shard(i, 3) for i in range(3)]
  assert sum(len(s) for s in shards) == len(m)
  assert {p.point_id for s in shards for p in s} == {p.point_id for p in m}
  pid = m.points[0].point_id
  (tmp_path / 'points' / pid).mkdir(parents=True)
  (tmp_path / 'points' / pid / 'result.json').write_text('{}')
  assert len(m.pending(str(tmp_path))) == len(m) - 1


def test_manifest_custom_file_names(tmp_path):
  m = Manifest([_spec().expand()[0]], manifest_name='m.jsonl', result_name='r.json')
  m.write(str(tmp_path))
  assert (tmp_path / 'm.jsonl').is_file() and not (tmp_path / 'manifest.jsonl').exists()
  reloaded = Manifest.load(str(tmp_path), manifest_name='m.jsonl', result_name='r.json')
  pid = reloaded.points[0].point_id
  (tmp_path / 'points' / pid).mkdir(parents=True)
  (tmp_path / 'points' / pid / 'r.json').write_text('{}')
  assert len(reloaded.pending(str(tmp_path))) == 0


# ---- metrics / schema --------------------------------------------------
def test_oblique_and_precision_metrics():
  rng = np.random.default_rng(0)
  ob = conifer.model.make_model(random_model(rng, n_trees=3, max_depth=4, n_features=6,
                                             split='oblique', oblique_k=3), CFG)
  mm = perf_metrics.get_model_metrics(ob)
  assert mm.oblique.fraction > 0 and mm.oblique.terms_max >= 2
  assert mm.precision.threshold.width == 18 and mm.precision.threshold.integer_bits == 8
  flat = mm.flatten()
  assert flat['oblique_fraction'] == mm.oblique.fraction
  assert flat['threshold_precision_width'] == 18.0 and flat['threshold_precision_int_bits'] == 8.0

  ax = conifer.model.make_model(random_model(rng, n_trees=3, max_depth=4, n_features=6), CFG)
  assert perf_metrics.get_model_metrics(ax).oblique.fraction == 0


def test_model_metrics_flatten_keeps_legacy_flat_names():
  # the shipped HLS estimators index get_model_metrics(...).flatten() by these exact names
  legacy_features = ['max_depth', 'n_trees', 'n_features', 'n_nodes', 'n_leaves',
                     'sparsity_mean', 'sparsity_std', 'sparsity_min', 'sparsity_max', 'sparsity_sum',
                     'sparsity_quartile_1', 'sparsity_quartile_3',
                     'feature_frequency_mean', 'feature_frequency_std', 'feature_frequency_min',
                     'feature_frequency_max', 'feature_frequency_sum',
                     'feature_frequency_quartile_1', 'feature_frequency_quartile_3']
  model = conifer.model.make_model(random_model(np.random.default_rng(0), n_trees=3, max_depth=3, n_features=4), CFG)
  assert set(legacy_features) <= set(perf_metrics.get_model_metrics(model).flatten())


def test_schema_versions_are_hardcoded_and_not_on_sub_dataclasses():
  model = conifer.model.make_model(random_model(np.random.default_rng(0), n_trees=2, max_depth=2, n_features=3), CFG)
  mm = perf_metrics.get_model_metrics(model)
  assert mm.schema_version == perf_metrics.MODEL_METRICS_SCHEMA_VERSION
  assert mm.flatten()['schema_version'] == perf_metrics.MODEL_METRICS_SCHEMA_VERSION
  for sub in (mm.sparsity, mm.feature_frequency, mm.oblique, mm.precision):
    assert not hasattr(sub, 'schema_version')

  sm = perf_metrics.SynthesisMetrics(outcome='ok', reason='')
  assert sm.schema_version == perf_metrics.SYNTHESIS_METRICS_SCHEMA_VERSION


def test_get_model_measurements_missing_output_dir(tmp_path):
  model = conifer.model.make_model(
    random_model(np.random.default_rng(0), n_trees=2, max_depth=2, n_features=3),
    _xilinxhls_cfg(tmp_path / 'never-written'))
  sm = perf_metrics.get_model_measurements(model, do_vsynth=True)
  assert sm.outcome == 'error' and 'output directory' in sm.reason


def test_get_model_measurements_no_reports(tmp_path):
  outdir = tmp_path / 'prj'
  outdir.mkdir()
  model = conifer.model.make_model(
    random_model(np.random.default_rng(0), n_trees=2, max_depth=2, n_features=3), _xilinxhls_cfg(outdir))
  sm = perf_metrics.get_model_measurements(model, do_vsynth=False)
  assert sm.outcome == 'hls_fail' and sm.reason == 'no HLS csynth report'


def test_get_model_measurements_malformed_report_does_not_raise(tmp_path):
  outdir = tmp_path / 'prj'
  report_dir = outdir / 'conifer_prj' / 'solution1' / 'syn' / 'report'
  report_dir.mkdir(parents=True)
  (report_dir / 'conifer_prj_csynth.xml').write_text('not xml')
  model = conifer.model.make_model(
    random_model(np.random.default_rng(0), n_trees=2, max_depth=2, n_features=3), _xilinxhls_cfg(outdir))
  sm = perf_metrics.get_model_measurements(model, do_vsynth=False)
  assert sm.outcome == 'hls_fail' and 'could not parse HLS report' in sm.reason


# ---- make_model / plan -----------------------------------------------
def test_make_model_applies_overrides(tmp_path):
  spec = ScanSpec(name='b', mode='grid',
                  base_config={'backend': 'xilinxhls', 'precision': 'ap_fixed<18,8>', 'clock_period': 5,
                               'xilinx_part': 'xcvu9p-flgb2104-2L-e', 'unroll': True},
                  axes=[Axis('n_trees', 'grid', values=[4]), Axis('max_depth', 'grid', values=[3]),
                        Axis('precision', 'choice', values=['ap_fixed<12,4>']),
                        Axis('clock_period', 'choice', values=[2.5])],
                  generator_defaults={'n_features': 5})
  point = spec.expand()[0]
  model = make_model(point, str(tmp_path), spec.base_config)
  assert model.config.threshold_precision == 'ap_fixed<12,4>'
  assert model.config.clock_period == 2.5


def test_resolve_sparsity_normal_defaults():
  from conifer.utils.performance.scan.build import _resolve_sparsity
  rng = np.random.default_rng(0)
  values = _resolve_sparsity({'dist': 'normal'}, rng, n_trees=2000)
  assert 0.35 < np.mean(values) < 0.65  # default mean 0.5, spread from default std 0.5


def test_dummy_cost_model_returns_constant():
  cm = DummyCostModel(seconds=123.0, mem_gb=2.0, disk_bytes=1000.0)
  a = cm.predict(_spec().expand()[0])
  b = cm.predict(_spec().expand()[1])
  assert a == b == (123.0, 2.0, 1000.0)


def test_group_into_batches_covers_all_points(tmp_path):
  Manifest.from_spec(_spec()).write(str(tmp_path))
  summary = group_into_batches(str(tmp_path), n_batches=3)
  ids = set()
  for shard in summary['shards']:
    ids |= {json.loads(line)['point_id'] for line in open(shard)}
  assert ids == {p.point_id for p in Manifest.load(str(tmp_path))}
