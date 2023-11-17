import conifer
from conifer.utils.metrics import array_summary_statistics, get_sparsity_metrics, get_feature_frequency_metrics
import numpy as np
import math
import os
import shutil
import pandas
import contextlib
import joblib
from multiprocessing import Pool
from tqdm import tqdm
from tabulate import tabulate
import random
from enum import Enum
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)

class Node:
  def __init__(self, i):
    self.i = i
    self.threshold = 0
    self.feature = -2
    self.value = 0
    self.child_left = None
    self.child_right = None
  def add_children(self, cl, cr):
    self.child_left = cl
    self.child_right = cr
  def __repr__(self):
    cl = self.child_left.i if not self.child_left is None else 'None'
    cr = self.child_right.i if not self.child_right is None else 'None'
    return f'Node({self.i}) (L : {cl}, R : {cr})'
  def is_leaf(self):
    return True if (self.child_left is None) and (self.child_right is None) else False

class Tree:
  '''
  Class to create randomised trees
  '''
  def __init__(self, max_depth : int, n_features : int, sparsity : float = 0., max_threshold : float = 2**5, max_value : float = 2**5):
    self.max_depth = max_depth
    self.nodes = [Node(0)]
    # insert the decision nodes
    n = 1
    while len(self.nodes) < 2 ** max_depth - 1:
      n = self._add_children_one_node(n)
    # randomly choose a feature for each node
    feature = np.random.randint(0, n_features, 2**max_depth-1)
    for i, node in enumerate(self.nodes):
      node.feature = feature[i]
    # prune to desired sparsity
    self._prune_to_sparsity(sparsity)
    self._rationalize_node_indices()
    # set decision thresholds that are logically consistent
    self._randomise_thresholds(max_threshold)
    self._sort()
    # add leaf nodes
    self._add_leaves(max_value)

  def _add_children_one_node(self, n):
    '''
    Add children to the first node without children
    '''
    for node in self.nodes:
      if node.is_leaf():
        cl = Node(n)
        n += 1
        cr = Node(n)
        n += 1
        node.add_children(cl, cr)
        self.nodes.append(cl)
        self.nodes.append(cr)
        break
    return n

  def _randomise_thresholds(self, max_threshold=1.):
    self.nodes[0].threshold = np.random.uniform(low=-max_threshold, high=max_threshold)
    # this traversal needs to be depth first
    for i in range(1,self.n_nodes()):
      node = self.nodes[i]
      # get the parents of node i
      j = i
      i_parents = []
      parent = node
      while parent.i > 0:
        parent = self.get_node_parent(parent)[0]
        i_parents.append(parent)
      # get the parents of node i that use the same feature
      i_parents_same_f = [p for p in i_parents if p.feature == node.feature]
      # if no parents use the same feature, use full min,max range
      if len(i_parents_same_f) == 0:
        min_t, max_t = -max_threshold, max_threshold
      # if some parents used the same feature, make sure new threshold is logically valid
      else:
        # determine whether node is to the left or right of the last parent using the same feature
        last_parent_same_f = i_parents_same_f[0]
        cl, cr = last_parent_same_f.child_left, last_parent_same_f.child_right
        p = node
        counter = 0
        while (not p in [cl, cr]) and counter < 10:
          p = self.get_node_parent(p)
          counter += 1
        # if node is to left of last parent using same feature, threshold must be lower
        if p == cl:
          min_t, max_t = -max_threshold, last_parent_same_f.threshold
        # if node is to right of last parent using same feature, threshold must be greater
        else:
          min_t, max_t = last_parent_same_f.threshold, max_threshold
      #if self.children_left[i] != -2:
      node.threshold = np.random.uniform(low=min_t, high=max_t)

  def _prune_to_sparsity(self, sparsity : float):
    '''
    Remove nodes until reaching the desired sparsity or only one node
    '''
    while self.sparsity() < sparsity and self.n_nodes() >= 2:
      ileaves = np.array([i for i in range(self.n_nodes()) if self.nodes[i].is_leaf()])
      iPrune = np.random.choice(ileaves, 1)[0]
      node = self.nodes[iPrune]
      self._remove_node(node)

  def _remove_node(self, node : Node):
    '''
    Remove the node from list of nodes and rewire the parent
    '''
    parent, side = self.get_node_parent(node)
    # remove the node from the list
    self.nodes.remove(node)
    # remove the node from the parent's children
    if side == 'left':
      parent.child_left = None
    else:
      parent.child_right = None
    
  def _rationalize_node_indices(self):
    '''
    Make all node indices contiguous
    '''
    indices = np.array([node.i for node in self.nodes])
    valid_indices = np.array(list(range(self.n_nodes())))
    invalid_indices = valid_indices[~np.isin(indices, valid_indices)]
    available_indices = valid_indices[~np.isin(valid_indices, indices)]
    for i, idx in enumerate(invalid_indices):
      self.nodes[idx].i = available_indices[i]

  def _sort(self):
    '''
    Sort by node index
    '''
    self.nodes = sorted(self.nodes, key = lambda n : n.i)

  def _add_leaves(self, max_value : float):
    '''
    Add leaf nodes
    '''
    def make_leaf(i):
      leaf = Node(i)
      leaf.feature = -2
      leaf.value = np.random.uniform(low=-max_value, high=max_value)
      leaf.threshold = 0
      return leaf
    
    n = self.n_nodes()
    n_nodes = self.n_nodes() # so that we don't add forever
    for i in range(n_nodes):
      node = self.nodes[i]
      if node.child_left is None:
        leaf = make_leaf(n)
        node.add_children(leaf, node.child_right)
        self.nodes.append(leaf)
        n += 1
      if node.child_right is None:
        leaf = make_leaf(n)
        node.add_children(node.child_left, leaf)
        self.nodes.append(leaf)
        n += 1

  def get_node_parent(self, node):
    for n in self.nodes:
      if node == n.child_left:
        return n, 'left'
      elif node == n.child_right:
        return n, 'right'

  def n_nodes(self):
    return len(self.nodes)
  
  def sparsity(self):
    return 1 - self.n_nodes() / (2**self.max_depth-1)
  
  @property
  def children_left(self):
    return [int(n.child_left.i) if not n.is_leaf() else -1 for n in self.nodes]
  @property
  def children_right(self):
    return [int(n.child_right.i) if not n.is_leaf() else -1 for n in self.nodes]
  @property
  def threshold(self):
    return [float(n.threshold) for n in self.nodes]
  @property
  def value(self):
    return [float(n.value) for n in self.nodes]
  @property
  def feature(self):
    return [int(n.feature) for n in self.nodes]
  
  def _to_dict(self):
    return {'children_left' : self.children_left,
            'children_right' : self.children_right,
            'feature' : self.feature,
            'threshold' : self.threshold,
            'value' : self.value,
           }

def make_model(n_trees: int, max_depth: int, n_features: int, sparsity : float = 0., xmax: int = 2**5):
    dictionary = {'max_depth' : max_depth, 'n_trees' : n_trees,
                  'n_features' : n_features,
                  'n_classes' : 2,
                  'init_predict' : [np.random.uniform(low=-xmax, high=xmax)],
                  'norm' : 1}
    
    def is_array_like(x):
        return hasattr(x, '__getitem__') or (hasattr(x, '__iter__') and hasattr(x, '__len__'))
    sparsity_per_tree = np.zeros(n_trees, dtype='float')
    if not is_array_like(sparsity):
      sparsity_per_tree += sparsity
      assert((sparsity < 1) and (sparsity >= 0)), f'sparsity must be in the interval [0, 1), but got lower & upper bounds: {sparsity}, {sparsity}'
    else:
      assert(len(sparsity) == n_trees), f'sparsity is an array, but has {len(sparsity)} elements where {n_trees} (n_trees) are needed'
      assert((max(sparsity) < 1) and (min(sparsity) >= 0)), f'sparsity must be in the interval [0, 1), but got lower & upper bounds: {min(sparsity)}, {max(sparsity)}'
      sparsity_per_tree = sparsity
    trees = []
    for i in range(n_trees):
      tree = Tree(max_depth, n_features, sparsity_per_tree[i], xmax, xmax)
      trees.append([tree._to_dict()]) # extra [] for classes
    dictionary['trees'] = trees
    return dictionary

class Experiment:
  '''
  Scan experiment interface
  '''
  n_experiments = 0
  def __init__(self,
               scandir : str,
               config : dict,
               n_trees : int,
               max_depth : int,
               n_features : int,
               xmax : float = 2.**5,
               sparsity : float = 0.):
    self.n_experiment = Experiment.n_experiments
    self.scandir = scandir
    self.config = config.copy()
    self.odir = f'{self.scandir}/prj_{self.n_experiment}'
    self.config['OutputDir'] = self.odir
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.n_features = n_features
    self.xmax = xmax
    self.sparsity = sparsity
    self.model = None
    Experiment.n_experiments += 1

  def make_model(self):
    if self.model is None:
      self.model = conifer.model.make_model(make_model(self.n_trees, self.max_depth, self.n_features, self.sparsity, self.xmax), self.config)
    return self.model

  def write(self):
    self.make_model()
    self.model.write()

  def save(self):
    self.make_model()
    self.model.save()

  def run(self, shrink=False):
    self.write()
    success = self.model.build()
    ret = None
    if success:
      ret = gather_report(self.odir)
    if shrink:
      self.shrink()
    return ret

  def get_estimator_features(self):
    self.make_model()
    x = [self.model.max_depth, self.model.n_nodes() - self.model.n_leaves(), self.model.n_trees,
         self.model.n_leaves()]
    sparsity_metrics = get_sparsity_metrics(self.model)
    for k, v in sparsity_metrics.items():
      x.append(v)
    return np.array([x])
  
  def estimate_build_time(self):
    estimator = get_build_time_model()
    return estimator.decision_function(self.get_estimator_features())

  def estimate_build_memory(self):
    estimator = get_build_memory_model()
    return estimator.decision_function(self.get_estimator_features())
  
  def estimate_disk_space(self):
    estimator = get_disk_space_model()
    return estimator.decision_function(self.get_estimator_features())
  
  def shrink(self):
    '''
    Remove all files except the model JSON, build reports and logs
    '''
    pn = self.model.config.project_name
    files_to_keep = [f'{pn}.json', 'build.log', 'vivado_build.log', 'vivado_synth.rpt', 'util.rpt', 'vivado.log',
                     'vitis_hls.log', 'vivado_hls.log', f'{pn}_csynth.rpt', f'{pn}_csynth.xml']
    for dirpath, dirnames, filenames in os.walk(self.odir):
      for f in filenames:
        if not f in files_to_keep:
          fp = os.path.join(dirpath, f)
          os.remove(fp)

class Scan:

  def __init__(self, scandir, experiments, shuffle=False):
    self.scandir = scandir
    self.experiments = experiments
    if shuffle:
      random.shuffle(self.experiments)

  def _summary_table(self, time, mem, disk):
    def format_bytes(bytes, decimal_places=2):
      for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
          if bytes < 1024.0:
              break
          bytes /= 1024.0
      return f"{bytes:.{decimal_places}f} {unit}", unit
    def format_time(seconds):
      days, remainder = divmod(seconds, 86400)
      hours, remainder = divmod(remainder, 3600)
      minutes, seconds = divmod(remainder, 60)

      time = ''
      if days > 0:
        time = f'{days} day{"s" if days > 1 else ""} '
      time += f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
      return time
    
    disk, disk_unit = format_bytes(disk)
    headers = ['Models', 'Jobs', 'Time', 'Peak Memory (GB)', f'Disk Space ({disk_unit})']
    table = [[len(self.experiments), format_time(time), mem, disk]]
    pretty_table = tabulate(table, headers=headers, tablefmt='orgtbl')
    return pretty_table
    
  def _print_scan_summary(self, i_experiments=None):
    if i_experiments is None:
      experiments = self.experiments
    else:
      experiments = [self.experiments[i] for i in i_experiments]
    time = np.sum([e.estimate_build_time() for e in experiments]) 
    mem = np.max([e.estimate_build_memory() for e in experiments])
    disk= np.sum([e.estimate_disk_space() for e in experiments])
    table = self._summary_table(time, mem, disk)

    table_len = int(np.ceil(len(table)/3))
    summary = '-' * table_len + '\n'
    summary += '| \N{evergreen tree} conifer performance scan summary \N{evergreen tree}\n'
    summary += f'| saving results to {self.scandir}\n'
    summary += '| Estimated performance:\n'
    summary += '-' * table_len + '\n'
    summary += table + '\n'
    summary += '-' * table_len + '\n'
    print(summary)
    return summary

  def write(self, njobs=1):
    filedir = os.path.basename(os.path.dirname(__file__))
    pool = Pool(njobs)
    r = list(tqdm(pool.imap(write_experiment, self.experiments), desc='\N{evergreen tree} writing scan ', total=len(self.experiments)))
    #shutil.copyfile(f'{filedir}/run_scan.sh', f'{self.scandir}/run_scan.sh')

  def save(self, njobs=1):
    filedir = os.path.basename(os.path.dirname(__file__))
    pool = Pool(njobs)
    r = list(tqdm(pool.imap(save_experiment, self.experiments), desc='\N{evergreen tree} writing scan ', total=len(self.experiments)))
    #shutil.copyfile(f'{filedir}/run_scan.sh', f'{self.scandir}/run_scan.sh')

def write_experiment(experiment):
  experiment.write()

def save_experiment(experiment):
  experiment.save()

def gather_report(prjdir):
  model = conifer.model.load_model(f'{prjdir}/my_prj.json')
  backend = model.config.backend
  fname = 'vivado_synth.rpt' if backend == 'xilinxhls' else 'util.rpt'
  
  rep = conifer.backends.common.read_vsynth_report(f'{prjdir}/{fname}')
  if backend == 'xilinxhls':
    hls_report = conifer.backends.common.read_hls_report(f'{prjdir}/my_prj_prj/solution1/syn/report/my_prj_csynth.xml')
    latency = hls_report['latency_best'] if hls_report is not None else 0
    build_log = conifer.backends.common.read_hls_log(f'{prjdir}/vitis_hls.log')
    build_time = build_log.get('time_seconds', None) if build_log is not None else 0
    build_mem = build_log.get('memory_GB', None) if build_log is not None else 0
  else:
    latency = 1 + model.max_depth + math.ceil(math.log2(model.n_trees))
    build_time = 0
    build_mem = 0
  results_d = {'latency' : latency, 'lut' : rep['lut'], 'ff' : rep['ff'],
               'max_depth' : model.max_depth, 'n_trees' : model.n_trees,
               'n_features' : model.n_features,
               'n_nodes' : model.n_nodes() - model.n_leaves(), 'n_leaves' : model.n_leaves(),
               'backend' : model.config.backend, 'build_time' : build_time, 'build_memory' : build_mem,
               'disk_space' : get_disk_size(prjdir)}
  results_d.update(get_sparsity_metrics(model))
  results_d.update(get_feature_frequency_metrics(model))
  return results_d

def gather_reports(scandir,):
  prjs = os.listdir(scandir)
  prjs = [prj for prj in prjs if 'prj' in prj and 
            (os.path.exists(f'{scandir}/{prj}/util.rpt') or os.path.exists(f'{scandir}/{prj}/vivado_synth.rpt'))]
  if len(prjs) == 0:
    logger.warn(f'Found no projects in {scandir}')
  else:
    logger.info(f'Found {len(prjs)} projects in {scandir}')

  results_l = []
  for prj in prjs:
    results_l.append(gather_report(f'{scandir}/{prj}/'))
  results_d = {}
  for k in results_l[0].keys():
    results_d[k] = []
  for result_d in results_l:
    for k in result_d.keys():
      results_d[k].append(result_d[k])
  return pandas.DataFrame(results_d)

def get_disk_size(start_path = '.'):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(start_path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      # skip if it is symbolic link
      if not os.path.islink(fp):
        total_size += os.path.getsize(fp)
  return total_size

model_build_time = None
model_build_memory = None
model_disk_space = None

def get_build_time_model():
  import conifer
  global model_build_time
  if model_build_time is None:
    model_build_time = conifer.model.load_model(f'{os.path.dirname(__file__)}/build_time.json')
  return model_build_time

def get_build_memory_model():
  import conifer
  global model_build_memory
  if model_build_memory is None:
    model_build_memory = conifer.model.load_model(f'{os.path.dirname(__file__)}/build_memory.json')
  return model_build_memory

def get_disk_space_model():
  import conifer
  global model_disk_space
  if model_disk_space is None:
    model_disk_space = conifer.model.load_model(f'{os.path.dirname(__file__)}/disk_space.json')
  return model_disk_space
