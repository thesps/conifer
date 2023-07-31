import conifer
import numpy as np
import math
import os
import shutil
import xml.etree.ElementTree as ET
import pandas
import argparse
import itertools
import yaml
import contextlib
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class Node:
  def __init__(self, i):
    self.i = i
  def add_children(self, cl, cr):
    self.child_left = cl
    self.child_right = cr
  def __repr__(self):
    cl = self.child_left.i if not self.is_leaf() else 'None'
    cr = self.child_right.i if not self.is_leaf() else 'None'
    return f'Node({self.i}) (L : {cl}, R : {cr})'
  def is_leaf(self):
    return False if getattr(self, 'child_left', False) else True

class Tree:
  '''
  Class to create randomised trees
  '''
  def __init__(self, depth):
    self.n = 0
    self.nodes = [Node(self.n)]
    self.n += 1
    while len(self.nodes) < 2 ** (depth + 1) - 1:
      self._add_children_one_node()

  def _add_children_one_node(self):
    '''
    Add children to the first node without children
    '''
    for node in self.nodes:
      if getattr(node, 'child_left', None) is None:
        cl = Node(self.n)
        self.n += 1
        cr = Node(self.n)
        self.n += 1
        node.add_children(cl, cr)
        self.nodes.append(cl)
        self.nodes.append(cr)
        break

  def children_left(self):
    return [n.child_left.i if not n.is_leaf() else -1 for n in self.nodes]

  def children_right(self):
    return [n.child_right.i if not n.is_leaf() else -1 for n in self.nodes]

  def feature(self):
    return [0 if not n.is_leaf() else -2 for n in self.nodes]
  
  def to_dict_randomise(self, max):
    return {'children_left' : self.children_left(),
            'children_right' : self.children_right(),
            'feature' : self.feature(),
            'threshold' : np.random.uniform(low=-max, high=max, size=len(self.nodes)).tolist(),
            'value' : np.random.uniform(low=-max, high=max, size=len(self.nodes)).tolist(),}

def make_model(n_trees: int, max_depth: int, max: int = 2**5):
    tree = Tree(max_depth)
    return {'max_depth' : max_depth, 'n_trees' : n_trees,
            'n_features' : 1,
            'n_classes' : 2, 'trees' : [[tree.to_dict_randomise(max=max)] for i in range(n_trees)],
            'init_predict' : [1],
            'norm' : 1}

def run_experiment(i, sl, N, max_depth, n_trees, repeat, config, scandir):
    odir = f'{scandir}/prj_n{n_trees}_d{max_depth}_{repeat}'
    config['OutputDir'] = odir
    model = conifer.model.make_model(make_model(n_trees, max_depth), config)
    model.write()
    model.build()

    report = model.read_report()
    if config['backend'] == 'xilinxhls':
      report['lut'] = report['vsynth']['lut']
      report['ff'] = report['vsynth']['ff']

    return {'latency'   : report['latency'],
            'lut'       : report['lut'],
            'ff'        : report['ff'],
            'max_depth' : max_depth,
            'n_trees'   : n_trees,
            'trial'     : repeat}

def run_scan(scandir: str,
         max_depths: "list[int]",
         n_trees: "list[int]",
         repeats: int,
         config: dict,
         jobs: int):
  repeats = list(range(repeats))
  experiments = list(itertools.product(max_depths, n_trees, repeats))
  sl = len(str(len(experiments))) # length of number of experiments string

  with tqdm_joblib(tqdm(desc="\N{evergreen tree} scan", total=len(experiments), ascii=" ▖▘▝▗▚▞█")) as progress_bar:
    results = joblib.Parallel(n_jobs=jobs)(joblib.delayed(run_experiment)(i, sl, len(experiments), max_depth, n_tree, repeat, config, scandir) for i, (max_depth, n_tree, repeat) in enumerate(experiments))

  reports = gather_reports(scandir, hls=config['backend']=='xilinxhls')
  r.to_csv(f'{scandir}/results.csv')

def gather_reports(scandir, hls=False):
  fname = 'vivado_synth.rpt' if hls else 'util.rpt'
  prjs = os.listdir(scandir)
  prjs = [prj for prj in prjs if 'prj' in prj and os.path.exists(f'{scandir}/{prj}/{fname}')]
  if len(prjs) == 0:
    logger.warn(f'Found no projects in {scandir}')
  else:
    logger.info(f'Found {len(prjs)} projects in {scandir}')
  results_d = {'latency' : [], 'lut' : [], 'ff' : [],
               'max_depth' : [], 'n_trees' : [], 'n_nodes' : [], 'n_leaves' : [], 'sparsity' : [],
               'trial' : [], 'build_time' : [], 'build_memory' : []}
  for prj in prjs:
    model = conifer.model.load_model(f'{scandir}/{prj}/my_prj.json')
    details = prj.split('_')
    trial = int(details[3])
    rep = conifer.backends.common.read_vsynth_report(f'{scandir}/{prj}/{fname}')
    if hls:
      latency = conifer.backends.common.read_hls_report(f'{scandir}/{prj}/my_prj_prj/solution1/syn/report/my_prj_csynth.xml')['latency_best']
      build_log = conifer.backends.common.read_hls_log(f'{scandir}/{prj}/vitis_hls.log')
      build_time = build_log['time_seconds']
      build_mem = build_log['memory_GB']
    else:
      latency = 1 + model.max_depth + math.ceil(math.log2(model.n_trees))
      build_time = 0
      build_mem = 0
    results_d['latency'].append(latency)
    results_d['lut'].append(rep['lut'])
    results_d['ff'].append(rep['ff'])
    results_d['n_trees'].append(model.n_trees)
    results_d['max_depth'].append(model.max_depth)
    results_d['n_nodes'].append(model.n_nodes())
    results_d['n_leaves'].append(model.n_leaves())
    results_d['sparsity'].append(model.sparsity())
    results_d['trial'].append(trial)
    results_d['build_time'].append(build_time)
    results_d['build_memory'].append(build_mem)
  return pandas.DataFrame(results_d)

def f_resources(X, k0, k1, k2):
  n_trees, n_nodes, max_depth = X
  return k0 + k1 * n_trees + k2 * n_nodes

def f_latency(X, k0, k1, k2):
  n_trees, n_nodes, max_depth = X
  return k0 + k1 * max_depth + k2 * np.log2(n_trees)

def do_fits(results: pandas.DataFrame):
  import pandas
  from scipy.optimize import curve_fit
  d = results
  k_lut, _ = curve_fit(f_resources, (d.n_trees, d.n_nodes - d.n_leaves, d.max_depth), d.lut)
  k_ff, _ = curve_fit(f_resources, (d.n_trees, d.n_nodes - d.n_leaves, d.max_depth), d.ff)
  k_lat, _ = curve_fit(f_latency, (d.n_trees, d.n_nodes - d.n_leaves, d.max_depth), d.latency)
  return {'lut' : k_lut, 'ff' : k_ff, 'latency' : k_lat}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--max-depth', type=int, nargs='+', default=[2, 3, 4])
  parser.add_argument('-n', '--n_trees', type=int, nargs='+', default=[10, 20, 30])
  parser.add_argument('-r', '--repeats', type=int, default=5)
  parser.add_argument('-o', '--out_dir', type=str, default='scan')
  parser.add_argument('-c', '--config', type=str)
  parser.add_argument('-j', '--jobs', type=int, default=1)
  args = parser.parse_args()
  with open(args.config) as f:
    config = yaml.safe_load(f)
  run_scan(args.out_dir, args.max_depth, args.n_trees, args.repeats, config, args.jobs)
