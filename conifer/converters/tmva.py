import xml.etree.ElementTree as ET
import numpy as np

def getOptionValue(bdt, optionName):
    for option in bdt.getroot().find('Options') :
        if optionName == option.get('name'):
            return option.text

def convert(bdt):
  max_depth = int(getOptionValue(bdt, 'MaxDepth'))
  n_trees = int(getOptionValue(bdt, 'NTrees'))
  n_features = int(bdt.find('Variables').attrib['NVar'])
  n_classes = int(bdt.find('Classes').attrib['NClass'])
  BoostType = str(getOptionValue(bdt, 'BoostType'))
  ensembleDict = {'max_depth' : max_depth, 'n_trees' : n_trees,
                  'n_features' : n_features,
                  'n_classes' : n_classes, 'trees' : [],
                  'init_predict' : [0.],
                  'norm' : 0, 'boost_type' : BoostType}
  for trees in bdt.find('Weights'):
    treesl = []
    #for tree in trees:
    # TODO find out how TMVA implements multi-class
    tree = trees
    weight = float(tree.attrib['boostWeight'])
    tree = treeToDict(bdt, tree)
    treesl.append(tree)
    ensembleDict['trees'].append(treesl)
    ensembleDict['norm'] += weight
  if BoostType == 'Grad':
    ensembleDict['norm'] = 1.
  else:
    # Invert the normalisation so FPGA can do '*' instead of '/'
    ensembleDict['norm'] = 1. / ensembleDict['norm'] 
  return ensembleDict


def recurse(node):
    yield node
    if len(node.getchildren()) > 0:
        for n in node.getchildren():
            for ni in recurse(n):
                yield ni

def treeToDict(bdt, tree):
  feature = []
  threshold = []
  value = []
  children_left = []
  children_right = []
  rootnode = tree[0]
  useYesNoLeaf = bool(getOptionValue(bdt, 'UseYesNoLeaf'))
  BoostType = str(getOptionValue(bdt, 'BoostType'))
  # In the fast pass add an ID
  for i, node in enumerate(recurse(rootnode)):
      node.attrib['ID'] = i
      attrib = node.attrib
      f = int(attrib['IVar']) if int(attrib['IVar']) != -1 else -2 # TMVA uses -1 for leaf, scikit-learn uses -2
      t = float(attrib['Cut'])
      if BoostType == 'Grad':
        v = float(attrib['res'])
      else:
        vPurity = float(attrib['purity']) * float(tree.attrib['boostWeight'])
        vType = float(attrib['nType']) * float(tree.attrib['boostWeight'])
        v = vType if useYesNoLeaf else vPurity
      feature.append(f)
      threshold.append(t)
      value.append(v)

  # Now add the children left / right reference
  for i, node in enumerate(recurse(rootnode)):
      ch = node.getchildren()
      if len(ch) > 0:
          # Swap the order of the left/right child depending on cut type attribute
          if bool(int(node.attrib['cType'])):
            l = ch[0].attrib['ID']
            r = ch[1].attrib['ID']
          else:
            l = ch[1].attrib['ID']
            r = ch[0].attrib['ID']
      else:
          l = -1
          r = -1
      children_left.append(l)
      children_right.append(r)

  treeDict = {'feature' : feature, 'threshold' : threshold, 'value' : value, 'children_left' : children_left, 'children_right' : children_right}

  return treeDict

