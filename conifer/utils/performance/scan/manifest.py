'''
Explicit, resumable list of scan points expanded from a ScanSpec.
Everything downstream (build, run, gather, plan) is keyed on Point.point_id.
'''
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_NAME = 'manifest.jsonl'
DEFAULT_SPEC_NAME = 'spec.yml'
DEFAULT_RESULT_NAME = 'result.json'


def _canonical(obj):
  '''Stable JSON string for hashing.'''
  return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


@dataclass
class Point:
  '''A single model to generate and synthesize, with a unique identity.'''
  name: str
  params: dict
  trial: int
  seed: int
  point_id: str = field(default=None)

  @classmethod
  def create(cls, name, params, trial, seed):
    pid = hashlib.sha1(_canonical([name, params, trial, seed]).encode()).hexdigest()[:12]
    return cls(name=name, params=dict(params), trial=trial, seed=seed, point_id=pid)

  def rng_seed(self):
    '''Deterministic generator seed derived from the point identity.'''
    return int(self.point_id, 16)

  def to_dict(self):
    return {'name': self.name, 'point_id': self.point_id, 'trial': self.trial,
            'seed': self.seed, 'params': self.params}

  @classmethod
  def from_dict(cls, d):
    return cls(name=d['name'], params=d['params'], trial=d['trial'],
               seed=d['seed'], point_id=d['point_id'])


def points_dir(root):
  return os.path.join(root, 'points')


def point_dir(root, point_id):
  return os.path.join(points_dir(root), point_id)


def result_path(root, point_id, result_name=DEFAULT_RESULT_NAME):
  return os.path.join(point_dir(root, point_id), result_name)


def has_result(root, point_id, result_name=DEFAULT_RESULT_NAME):
  return os.path.isfile(result_path(root, point_id, result_name))


class Manifest:
  '''An ordered collection of Points, backed by a manifest.jsonl file.'''
  def __init__(self, points, spec=None, manifest_name=DEFAULT_MANIFEST_NAME,
               spec_name=DEFAULT_SPEC_NAME, result_name=DEFAULT_RESULT_NAME):
    self.points = list(points)
    self.spec = spec
    self.manifest_name = manifest_name
    self.spec_name = spec_name
    self.result_name = result_name

  def __len__(self):
    return len(self.points)

  def __iter__(self):
    return iter(self.points)

  @classmethod
  def from_spec(cls, spec, **kwargs):
    return cls(spec.expand(), spec=spec, **kwargs)

  def write(self, root):
    '''Write manifest.jsonl (and spec.yml when available) under root.'''
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, self.manifest_name), 'w') as f:
      for p in self.points:
        f.write(_canonical(p.to_dict()) + '\n')
    if self.spec is not None:
      self.spec.to_yaml(os.path.join(root, self.spec_name))

  @classmethod
  def load(cls, root, manifest_name=DEFAULT_MANIFEST_NAME, result_name=DEFAULT_RESULT_NAME):
    '''Load a manifest from a directory or a manifest.jsonl path.'''
    path = root if os.path.isfile(root) else os.path.join(root, manifest_name)
    with open(path) as f:
      points = [Point.from_dict(json.loads(line)) for line in f if line.strip()]
    return cls(points, manifest_name=manifest_name, result_name=result_name)

  def shard(self, index, n_shards):
    '''Return the round-robin sub-manifest index/n_shards (0-based index).'''
    if not 0 <= index < n_shards:
      raise ValueError(f'shard index {index} out of range for {n_shards} shards')
    return Manifest(self.points[index::n_shards], spec=self.spec, manifest_name=self.manifest_name,
                    spec_name=self.spec_name, result_name=self.result_name)

  def pending(self, root):
    '''Return the sub-manifest of points that have no result yet.'''
    pending_points = [p for p in self.points if not has_result(root, p.point_id, self.result_name)]
    return Manifest(pending_points, spec=self.spec, manifest_name=self.manifest_name,
                    spec_name=self.spec_name, result_name=self.result_name)


def expand_spec(spec_path, scandir):
  '''Load a ScanSpec, expand it and write manifest + spec into scandir.'''
  from conifer.utils.performance.scan.spec import ScanSpec
  manifest = Manifest.from_spec(ScanSpec.load(spec_path))
  manifest.write(scandir)
  logger.info(f'{len(manifest)} point scan expanded to: {os.path.join(scandir, manifest.manifest_name)}')
  return manifest
