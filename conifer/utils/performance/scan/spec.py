'''
Declarative, serialisable description of a performance scan.
A ScanSpec expands deterministically into an explicit list of scan points.
'''
import itertools
import json
from dataclasses import dataclass, field, asdict

import numpy as np

# config keys routed to the conifer backend config; every other param is a random_model kwarg
CONFIG_KEYS = {'precision', 'input_precision', 'threshold_precision', 'weight_precision',
               'score_precision', 'xilinx_part', 'clock_period', 'unroll'}
_AXIS_KINDS = ('grid', 'choice', 'randint', 'randfloat')


@dataclass
class Axis:
  '''One scanned parameter and how it is sampled.'''
  name: str
  kind: str = 'grid'
  values: list = None          # grid / choice
  low: float = None            # randint / randfloat (inclusive low)
  high: float = None           # randint / randfloat (randint high is inclusive)
  log: bool = False            # randfloat: sample in log space

  def __post_init__(self):
    if self.kind not in _AXIS_KINDS:
      raise ValueError(f'axis {self.name}: kind must be one of {_AXIS_KINDS}, got {self.kind!r}')
    if self.kind in ('grid', 'choice') and not self.values:
      raise ValueError(f'axis {self.name}: {self.kind} axis needs non-empty values')
    if self.kind in ('randint', 'randfloat') and (self.low is None or self.high is None):
      raise ValueError(f'axis {self.name}: {self.kind} axis needs low and high')

  def grid_values(self):
    '''Values enumerated for this axis in grid mode.'''
    if self.kind in ('grid', 'choice'):
      return list(self.values)
    if self.kind == 'randint':
      return list(range(int(self.low), int(self.high) + 1))
    raise ValueError(f'axis {self.name}: kind {self.kind} cannot be enumerated in grid mode')

  def sample(self, rng):
    '''Draw one value for this axis in random mode.'''
    if self.kind in ('grid', 'choice'):
      return self.values[int(rng.integers(0, len(self.values)))]
    if self.kind == 'randint':
      return int(rng.integers(int(self.low), int(self.high) + 1))
    if self.log:
      return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
    return float(rng.uniform(self.low, self.high))


@dataclass
class ScanSpec:
  '''A named scan: axes, sampling mode, trials, base config and generator defaults.'''
  name: str
  axes: list = field(default_factory=list)
  mode: str = 'grid'                       # 'grid' or 'random'
  n_samples: int = None                    # random mode: number of parameter combinations
  n_trials: int = 1                        # repeats per combination (fresh random model each)
  seed: int = 0
  base_config: dict = field(default_factory=dict)
  generator_defaults: dict = field(default_factory=dict)

  def __post_init__(self):
    self.axes = [a if isinstance(a, Axis) else Axis(**a) for a in self.axes]
    if self.mode not in ('grid', 'random'):
      raise ValueError(f"mode must be 'grid' or 'random', got {self.mode!r}")
    if self.mode == 'random' and not self.n_samples:
      raise ValueError('random mode needs n_samples')

  # ---- serialisation -------------------------------------------------------
  def to_dict(self):
    return asdict(self)

  @classmethod
  def from_dict(cls, d):
    return cls(**d)

  def to_json(self, path):
    with open(path, 'w') as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def from_json(cls, path):
    with open(path) as f:
      return cls.from_dict(json.load(f))

  def to_yaml(self, path):
    import yaml
    with open(path, 'w') as f:
      yaml.safe_dump(self.to_dict(), f, sort_keys=False)

  @classmethod
  def from_yaml(cls, path):
    import yaml
    with open(path) as f:
      return cls.from_dict(yaml.safe_load(f))

  @classmethod
  def load(cls, path):
    '''Load a spec from .yaml/.yml or .json by extension; raises on any other extension.'''
    path = str(path)
    if path.endswith(('.yml', '.yaml')):
      return cls.from_yaml(path)
    if path.endswith('.json'):
      return cls.from_json(path)
    raise ValueError(f'unrecognised spec file extension: {path!r} (expected .yml/.yaml or .json)')

  # ---- expansion --------------------------------------------------------
  def _combinations(self):
    '''Yield the ordered dict of axis values for each parameter combination.'''
    if self.mode == 'grid':
      grids = [a.grid_values() for a in self.axes]
      for combo in itertools.product(*grids):
        yield {a.name: v for a, v in zip(self.axes, combo)}
    else:
      rng = np.random.default_rng(self.seed)
      for _ in range(self.n_samples):
        yield {a.name: a.sample(rng) for a in self.axes}

  def expand(self):
    '''Return the explicit list of Point objects for this spec.'''
    from conifer.utils.performance.scan.manifest import Point
    points = []
    for combo in self._combinations():
      params = {**self.generator_defaults, **combo}
      for trial in range(self.n_trials):
        points.append(Point.create(self.name, params, trial, self.seed))
    return points
