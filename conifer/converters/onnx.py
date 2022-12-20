import numpy as np
import math

#main converter function
def convert_bdt(onnx_clf):
  treelist,max_depth,base_values,no_features,no_classes=convert_graph(onnx_clf)
  ensembleDict = {'max_depth' : max_depth, 'n_trees' : len(treelist),
                   'trees' : [],'n_features' : no_features,
                  'n_classes' : no_classes,
                  'init_predict' : base_values,
                  'norm' : 1}
  for trees in treelist:
    treesl = []
    for treeDict in trees:
      for key in treeDict.keys():
        treeDict[key]=treeDict[key].tolist()
      tree = treeDict
      # NB node values are multiplied by the learning rate here, saving work in the FPGA
      tree['value'] = (np.array(tree['value']) * 1.0).tolist()
      treesl.append(tree)
    ensembleDict['trees'].append(treesl)

  return ensembleDict

def convert(onnx_clf):
    return convert_bdt(onnx_clf)

def convert_graph(onnx_clf):
  if(onnx_clf.graph.node[1].name=='ZipMap'):
    no_classes=max(onnx_clf.graph.node[1].attribute[0].ints) +1
  elif(onnx_clf.graph.node[2].name=='ZipMap'):
    no_classes=max(onnx_clf.graph.node[2].attribute[0].ints) +1

  node = onnx_clf.graph.node[0]
  attr_dict={}
  for i, attribute in enumerate(node.attribute):
      attr_dict[attribute.name]=i

  n_estimators=max(node.attribute[attr_dict['nodes_treeids']].ints)+1

  #converting flat representtaion in to numpy arrays through key value relationship
  tree_ids=np.array(node.attribute[attr_dict['nodes_treeids']].ints)
  children_right=np.array(node.attribute[attr_dict['nodes_falsenodeids']].ints)
  children_left=np.array(node.attribute[attr_dict['nodes_truenodeids']].ints)
  threshold=np.array(node.attribute[attr_dict['nodes_values']].floats)
  feature=np.array(node.attribute[attr_dict['nodes_featureids']].ints)
  leaf_values=np.array(node.attribute[attr_dict['class_weights']].floats)
  node_values=np.array(node.attribute[attr_dict['nodes_values']].floats)
  modes=np.array(node.attribute[attr_dict['nodes_modes']].strings)
  values_copy=np.copy(leaf_values)
  tree_no=len(np.unique(tree_ids))
  treelist=[]
  max_childern=0

  #create tree dictionary items from onnx graphical representation using numpy array slicing
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
    dict_tree['value']=node_values[tree_ids==tree_id]
    no_leaf_nodes=np.count_nonzero(mode==b'LEAF')
    dict_tree['value'][mode==b'LEAF']=values_copy[:no_leaf_nodes]
    values_copy=np.delete(values_copy, np.arange(0,no_leaf_nodes))
    treelist.append(dict_tree)
    max_childern=max(max_childern,len(dict_tree['children_left']))


  #finding depth of tree through maximum number of childern in the left branch of tree
  max_depth=math.ceil(math.log2(max_childern)-1)

  #base values and total number of features are found through onnx representation
  base_values=list(node.attribute[attr_dict['base_values']].floats)
  no_features=onnx_clf.graph.input[0].type.tensor_type.shape.dim[1].dim_value
  no_features=onnx_clf.graph.input[0].type.tensor_type.shape.dim[1].dim_value
  treelist=np.array(treelist)

  #rearranging tree list arrays for binary or multiclass 
  if (no_classes>2):
    treelist=treelist.reshape(-1,no_classes)
  else:
    treelist=treelist.reshape(treelist.shape[0],1)

  return treelist, max_depth, base_values, no_features, no_classes
