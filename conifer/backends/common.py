from conifer.model import DecisionTreeBase, ConfigBase
import numpy as np
import xml.etree.ElementTree as ET
import re
import os

class BottomUpDecisionTree(DecisionTreeBase):
  _tree_fields = DecisionTreeBase._tree_fields + ['parent', 'depth', 'iLeaf']
  def __init__(self, treeDict):
    for key in DecisionTreeBase._tree_fields:
        val = treeDict.get(key, None)
        assert val is not None, f"Missing expected key {key} in treeDict"
        setattr(self, key, val)
    self.addParentAndDepth()

  def padTree(self, max_depth):
    '''Pad a tree with dummy nodes if not perfectly balanced or depth < max_depth'''
    n_nodes = len(self.children_left)
    # while th tree is unbalanced
    while n_nodes != 2 ** (max_depth + 1) - 1:
      for i in range(n_nodes):
        if self.children_left[i] == -1 and self.depth[i] != max_depth:
          self.children_left.extend([-1, -1])
          self.children_right.extend([-1, -1])
          self.parent.extend([i, i])
          self.feature.extend([-2, -2])
          self.threshold.extend([-2.0, -2.0])
          val = self.value[i]
          self.value.extend([val, val])
          newDepth = self.depth[i] + 1
          self.depth.extend([newDepth, newDepth])
          iRChild = len(self.children_left) - 1
          iLChild = iRChild - 1
          self.children_left[i] = iLChild
          self.children_right[i] = iRChild
      n_nodes = len(self.children_left)
    self.iLeaf = []
    for i in range(n_nodes):
      if self.depth[i] == max_depth:
        self.iLeaf.append(i)

  def addParentAndDepth(tree):
    n = len(tree.children_left) # number of nodes
    parents = [0] * n
    for i in range(n):
      j = tree.children_left[i]
      if j != -1:
        parents[j] = i
      k = tree.children_right[i]
      if k != -1:
        parents[k] = i
    parents[0] = -1
    tree.parent = parents
    # Add the depth info
    tree.depth = [0] * n
    for i in range(n):
      depth = 0
      parent = tree.parent[i]
      while parent != -1:
        depth += 1
        parent = tree.parent[parent]
      tree.depth[i] = depth

class MultiPrecisionConfig(ConfigBase):
  '''
  Class representing configuration with fields for separate input, threshold, and score precisions
  Precisions fall back on 'precision' if not specified
  '''
  _config_fields = ConfigBase._config_fields + ['input_precision', 'threshold_precision', 'score_precision']
  _mp_alts = {'input_precision'     : ['InputPrecision'],
              'threshold_precision' : ['ThresholdPrecision'],
              'score_precision'     : ['ScorePrecision']
              }
  _alternates = {**ConfigBase._alternates, **_mp_alts}
  _mp_defaults = {'precision'     : 'float'}
  _defaults = {**ConfigBase._defaults, **_mp_defaults}

  def __init__(self, configDict, validate=True):
    super(MultiPrecisionConfig, self).__init__(configDict, validate=False)
    precision = None
    for k in ['precision', 'Precision']:
      val = configDict.get(k, None)
      if val is not None:
          precision = val

    if getattr(self, 'input_precision', None) is None:
      self.input_precision = precision
    if getattr(self, 'threshold_precision', None) is None:
      if self.input_precision is not None:
        self.threshold_precision = self.input_precision
      else:
        self.threshold_precision = precision
    if getattr(self, 'score_precision', None) is None:
      self.score_precision = precision
    if validate:
      self._validate()

  def any_ap_types(self):
    return np.any(['ap_' in t for t in [self.input_precision, self.threshold_precision, self.score_precision]])

def read_hls_report(filename: str) -> dict:
  '''
  Extract estimated performance metrics from HLS C Synthesis such as:
    latency (min, max), interval (min, max), resources
  Parameters
  ----------
  filename : string
    Name of XML HLS report file
  Returns
  ----------
  dictionary of extracted report contents
  '''
  
  if os.path.exists(filename):
    report = {}
    xml = ET.parse(filename)
    PE = xml.find('PerformanceEstimates')
    if PE is not None:
      SoOL = PE.find('SummaryOfOverallLatency')
      if SoOL is not None:
        report['latency_best'] = SoOL.find('Best-caseLatency')
        report['latency_worst'] = SoOL.find('Worst-caseLatency')
        report['interval_best'] = SoOL.find('Interval-min')
        report['interval_worst'] = SoOL.find('Interval-max')
    AE = xml.find('AreaEstimates')
    if AE is not None:
      R = AE.find('Resources')
      if R is not None:
        report['lut'] = R.find('LUT')
        report['ff'] = R.find('FF')
        report['dsp'] = R.find('DSP')
        report['bram18'] = R.find('BRAM_18K')

    for key in report.keys():
      if key is not None:
        report[key] = int(report[key].text)
    return report
  else:
    return None

def read_hls_log(filename: str) -> dict:
  '''
  Extract build metrics from HLS C Synthesis log such as:
    synthesis time, synthesis memory usage
  Parameters
  ----------
  filename : string
    Name of HLS log file
  Returns
  ----------
  dictionary of extracted log contents
  '''
  if os.path.exists(filename):
    report = {}
    f = open(filename, 'r')
    for line in f.readlines():
      if 'HLS 200-112' in line: # build summary line
        search = 'Total elapsed time: ([0-9]+)\.*([0-9]*) seconds'
        m = re.search(search, line)
        if m is not None:
          report['time_seconds'] = float(m.group(1))
          if m.group(2) != '':
            report['time_seconds'] += float(m.group(2))/100
        else:
          report['time_seconds'] = None
        search = 'peak allocated memory: ([0-9]+)\.*([0-9]*) ([k,M,G])B'
        m = re.search(search, line)
        if m is not None:
          mem = float(m.group(1))
          if m.group(2) != '':
            mem += float(m.group(2))/1000
          div = 1
          if m.group(3) == 'G':
            div = 1
          elif m.group(3) == 'M':
            div=1e3
          elif m.group(3) == 'k':
            div=1e6
          report['memory_GB'] = mem / div
        else:
          report['memory_GB'] = None
          report['time_seconds'] = None
    return report
  else:
    return None

def read_vsynth_report(filename):
  section = 0
  if os.path.exists(filename):
    report = {}
    f = open(filename, 'r')
    for line in f.readlines():
      # track which report section the line is in for filtering
      if '1. CLB Logic' in line:
        section = 1
      elif '1.1 Summary of Registers by Type' in line:
        section = 1.1
      elif '2. BLOCKRAM' in line:
        section = 2
      elif '3. ARITHMETIC' in line:
        section = 3
      elif '4. I/O' in line:
        section = 4

      # extract the value from the tables in each section
      if section == 1 and 'CLB LUTs*' in line:
        report['lut'] = int(line.split('|')[2])
      elif section == 1 and 'CLB Registers' in line:
        report['ff'] = int(line.split('|')[2])
      elif section == 2 and 'RAMB18' in line and 'Note' not in line:
        report['bram18'] = int(line.split('|')[2])
      elif section == 3 and 'DSPs' in line:
        report['dsp'] = int(line.split('|')[2])
    return report
  else:
    return None
