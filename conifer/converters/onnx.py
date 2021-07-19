import numpy as np
from .converter import addParentAndDepth, padTree
from ..model import model
import math

def convert_bdt(onnx_clf):
  treelist,max_depth=convert_graph(onnx_clf)
  ensembleDict = {'max_depth' : max_depth, 'n_trees' : len(treelist),
                   'trees' : [],
                  #'init_predict' : bdt._raw_predict_init(np.zeros(bdt.n_features_).reshape(1, -1))[0].tolist(),
                  'norm' : 1}
  #for trees in treelist:
   # treesl = []
  for treeDict in treelist:
    treeDict = ParentandDepth(treeDict)
    tree = padTree(ensembleDict, treeDict)
    # NB node values are multiplied by the learning rate here, saving work in the FPGA
    tree['value'] = (np.array(tree['value'])[:,0,0] * 1).tolist()
    
  ensembleDict['trees'].append(trees)

  return ensembleDict


def convert(onnx_clf):
    #print(onnx_clf)
    #if onnx_clf.graph.node[0]=='TreeEnsembleClassifier'
    return convert_bdt(onnx_clf)
    #elif 'RandomForest' in bdt.__class__.__name__:
    #    return convert_random_forest(onnx_clf)

def get_key(val,attr_dict):
      for key, value in attr_dict.items():
           if val == value:
               return key
      return "key doesn't exist"

def convert_graph(onnx_clf):
  node = onnx_clf.graph.node[0]
  attr_dict={}
  key=0
  for attribute in node.attribute:
      attr_dict[key]=attribute.name
      key=key+1
  print(attr_dict)
  print("\n\n")

  n_estimators=max(node.attribute[get_key('nodes_treeids',attr_dict)].ints)+1
  print(n_estimators)

  tree_ids=np.array(node.attribute[get_key('nodes_treeids',attr_dict)].ints).tolist()
  children_right=np.array(node.attribute[get_key('nodes_falsenodeids',attr_dict)].ints).tolist()
  children_left=np.array(node.attribute[get_key('nodes_truenodeids',attr_dict)].ints).tolist()
  threshold=np.array(node.attribute[get_key('nodes_values',attr_dict)].floats).tolist()
  feature=np.array(node.attribute[get_key('nodes_featureids',attr_dict)].ints).tolist()
  values=np.array(node.attribute[get_key('class_weights',attr_dict)].floats).tolist()
  modes=np.array(node.attribute[get_key('nodes_modes',attr_dict)].strings).tolist()
  values_copy=np.copy(values)
  print("\n\nUnique Nodes_treeids",np.unique(tree_ids))
  tree_no=len(np.unique(tree_ids))
  print("Number of trees",tree_no)
  #treelist = [dict() for x in range(tree_no)]
  #print(treelist)
  treelist=[]
  max_childern=0

  #print(children_left[tree_ids==0])
  for tree_id in np.unique(tree_ids):
          dict_tree={}
          mode=modes[tree_ids==tree_id]
          dict_tree['children_left']=children_left[tree_ids==tree_id]
          dict_tree['children_right']=children_right[tree_ids==tree_id]
          dict_tree['feature']=feature[tree_ids==tree_id]
          dict_tree['threshold']=threshold[tree_ids==tree_id]
          dict_tree['feature'][mode==b'LEAF'] = -2
          dict_tree['threshold'][mode==b'LEAF'] = -2
          dict_tree['children_left'][mode==b'LEAF'] = -1
          dict_tree['children_right'][mode==b'LEAF'] = -1
          no_leaf_nodes=np.count_nonzero(mode==b'LEAF')
          dict_tree['values']=values_copy[:no_leaf_nodes]
          values_copy=np.delete(values_copy, np.arange(0,no_leaf_nodes))
          treelist.append(dict_tree)
          max_childern=max(max_childern,len(dict_tree['children_left']))
  print(treelist)
  max_depth=math.ceil(math.log2(max_childern)-1)
  print(max_depth)
  return treelist, max_depth

def ParentandDepth(treeDict):
  # Extract the relevant tree parameters
  treeDict = addParentAndDepth(treeDict)
  return treeDict



  ##flat representation
  ##scikit-learn representation --> there is an array of trees with different attributes
  ##ONNX --> one array per attribute for the whole model 
  ##nodes_treeids --> the index of the tree --> flattened over all trees
  ##no estimator loop
  ##
  
