'''
Pre-flight cost estimate for a scan and greedy (LPT) grouping of points into balanced shards.
Costs come from a CostModel; DummyCostModel is a constant placeholder until a real one is trained
on the (auto-versioned, see schema.py) result schema alongside the latency/LUT/FF predictors.
'''
import json
import math
import os

import numpy as np

from conifer.utils.performance.scan.manifest import Manifest


class DummyCostModel:
  '''Placeholder: predicts the same (seconds, mem_gb, disk_bytes) for every point.'''
  def __init__(self, seconds=300.0, mem_gb=4.0, disk_bytes=5e7):
    self.seconds, self.mem_gb, self.disk_bytes = seconds, mem_gb, disk_bytes

  def predict(self, point):
    return self.seconds, self.mem_gb, self.disk_bytes


def _point_costs(manifest, cost_model):
  '''Return arrays of predicted (seconds, mem_gb, disk_bytes) for every point in the manifest.'''
  rows = [cost_model.predict(p) for p in manifest]
  secs, mem, disk = (np.array([r[i] for r in rows], dtype=float) for i in range(3))
  return secs, mem, disk


def plan(scandir, cost_model=None):
  '''Print a cost summary for a scan and return a per-point DataFrame of predictions.'''
  import pandas as pd
  cost_model = cost_model or DummyCostModel()
  manifest = Manifest.load(scandir)
  secs, mem, disk = _point_costs(manifest, cost_model)
  df = pd.DataFrame({'point_id': [p.point_id for p in manifest],
                     'pred_time_s': secs, 'pred_mem_gb': mem, 'pred_disk_bytes': disk})
  q = np.quantile(secs, [0.5, 0.9, 1.0]) if len(secs) else [0, 0, 0]
  print(f'points            : {len(manifest)}')
  print(f'total CPU-hours    : {secs.sum() / 3600:.1f}  (1 core)')
  print(f'per-point seconds  : median {q[0]:.0f}  p90 {q[1]:.0f}  max {q[2]:.0f}')
  print(f'peak memory (GB)   : {mem.max() if len(mem) else 0:.1f}')
  print(f'total disk (GB)    : {disk.sum() / 1024 ** 3:.1f}  (before shrink)')
  if isinstance(cost_model, DummyCostModel):
    print('cost source        : placeholder DummyCostModel, no trained cost model yet')
  return df


def group_into_batches(scandir, n_batches=None, target_hours=None, cost_model=None):
  '''Balance points across shards by predicted cost (LPT) and write shards/shard_NN.jsonl.'''
  cost_model = cost_model or DummyCostModel()
  manifest = Manifest.load(scandir)
  secs, _, _ = _point_costs(manifest, cost_model)
  if n_batches is None:
    if not target_hours:
      raise ValueError('provide n_batches or target_hours')
    n_batches = max(1, math.ceil(secs.sum() / 3600 / target_hours))

  order = np.argsort(secs)[::-1]
  loads = np.zeros(n_batches)
  batches = [[] for _ in range(n_batches)]
  for idx in order:
    b = int(np.argmin(loads))
    batches[b].append(manifest.points[idx])
    loads[b] += secs[idx]

  shard_dir = os.path.join(scandir, 'shards')
  os.makedirs(shard_dir, exist_ok=True)
  files = []
  for b, pts in enumerate(batches):
    fp = os.path.join(shard_dir, f'shard_{b:02d}.jsonl')
    with open(fp, 'w') as f:
      for p in pts:
        f.write(json.dumps(p.to_dict(), sort_keys=True) + '\n')
    files.append(fp)
  summary = {'n_batches': n_batches, 'batch_hours': [round(x / 3600, 2) for x in loads],
             'shards': files}
  json.dump(summary, open(os.path.join(shard_dir, 'plan.json'), 'w'), indent=2)
  print(f'{n_batches} shards, {min(loads) / 3600:.1f}-{max(loads) / 3600:.1f} h each -> {shard_dir}')
  return summary


def add_plan_arguments(parser):
  parser.add_argument('scandir')
  parser.add_argument('--batches', type=int, help='number of shards to write (LPT balanced)')
  parser.add_argument('--hours', type=float, help='target wall-clock hours per shard (sets --batches)')


def plan_from_args(args):
  plan(args.scandir)
  if args.batches or args.hours:
    group_into_batches(args.scandir, n_batches=args.batches, target_hours=args.hours)
