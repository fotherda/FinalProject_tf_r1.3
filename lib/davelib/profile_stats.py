'''
Created on 28 Jul 2017

@author: david
'''
import sys, os, logging, io, re
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter
from collections import defaultdict
from tensorflow.python.profiler.model_analyzer import _build_options, Profiler
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.core.profiler import tfprof_output_pb2
from davelib.utils import show_all_variables, RedirectStdStreams, stderr_redirector, \
  stdout_redirector, colour
from davelib.layer_name import LayerName, sort_func, ordered_layers
from matplotlib.patches import Rectangle

PERF_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 0,
    'min_float_ops': 1,
    'device_regexes': ['.*'],
    'order_by': 'cpu_micros',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': False,
    'select': ['float_ops','cpu_micros','bytes','peak_bytes','residual_bytes','output_bytes',
               'op_types','tensor_value','input_shapes','occurrence'],
    'viz': False,
    'dump_to_file': '',
#     'output': ''
#     'output': 'timeline:outfile=timeline_perf'
    'output': 'file:outfile=profile_perf_stats'
}

ALL_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 0,
    'min_float_ops': 0,
    'device_regexes': ['.*'],
    'order_by': 'cpu_micros',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': False,
    'select': ['params','float_ops','cpu_micros','bytes','peak_bytes','residual_bytes','output_bytes',
               'op_types','tensor_value','input_shapes','occurrence'],
    'viz': False,
    'dump_to_file': '',
#     'output': ''
#     'output': 'timeline:outfile=timeline_perf'
    'output': 'file:outfile=profile_perf_stats'
}


PERF_OPTIONS = copy.deepcopy(ALL_OPTIONS)
PERF_OPTIONS['min_float_ops'] = 1
PERF_OPTIONS['select'] = ['float_ops','cpu_micros','bytes','peak_bytes','residual_bytes',
                          'output_bytes', 'op_types','tensor_value','input_shapes','occurrence']
PARAM_OPTIONS = copy.deepcopy(ALL_OPTIONS)
PARAM_OPTIONS['min_params'] = 1
PARAM_OPTIONS['select'] = ['bytes','peak_bytes','residual_bytes','output_bytes','params',
                           'op_types','tensor_value','input_shapes','occurrence']

PERF_OPTIONS_PRINT = copy.deepcopy(PERF_OPTIONS)
PERF_OPTIONS_PRINT['output'] = ''
PARAM_OPTIONS_PRINT = copy.deepcopy(PARAM_OPTIONS)
PARAM_OPTIONS_PRINT['output'] = ''
ALL_OPTIONS_PRINT = copy.deepcopy(ALL_OPTIONS)
ALL_OPTIONS_PRINT['output'] = ''

def total_bytes_count(profile_stats):
  return profile_stats._scope_all_stats.total_peak_bytes + \
         profile_stats._scope_all_stats.total_output_bytes   
         

class NodeWrapper:
  def __init__(self, node, value, name=None):
    self._node = node
    if not name:
      name = node.name
    self._key = name 
#     self._key = re.sub(r"unit_(\d\b)", r"unit_0\1", name) #this makes the sorting work
    self._value = value

  def __key(self):
    return self._key
   
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      if not self.__key() and not other.__key():
        return True
      else:
        return self.__key() == other.__key()
    return False
  
  def __hash__(self):
    return hash(self.__key())
  
  def __lt__(self, other):
    if not self:
      if not other:
        return True
      else:
        return True
    elif not other:
      return False
    else:
      return self._key < other._key




def get_relevant_nodes(node, attrname, print_nodes=False):
  relevant_children = []
  _flatten(node, attrname, relevant_children)
  total = 0
  wrapped_list = []
  for child in relevant_children:
    value = getattr(child, attrname)
    total += value
    wrapped_list.append(NodeWrapper(child,value))
    if print_nodes:
      print('%d\t%s'%(value, child.name))
  return wrapped_list, total
    
DEFAULT = -1

def _flatten(node, attrname, relevant_children):
  attr = getattr(node, attrname, DEFAULT) #node with this attribute
  if attr!=DEFAULT and attr!=0: #node with this attribute
    relevant_children.append(node)

  if hasattr(node,'children'):
    children = getattr( node, 'children' )
    for child in children:
      _flatten(child, attrname, relevant_children)


class ProfileStats(object):
  '''
  holds profiling information for a network
  '''

  def __init__(self, run_metadata_list, graph, net_desc=None):
    #these can be pickled
    self._net_desc = net_desc
    self.serialize_data(graph, run_metadata_list)
    #these can't be pickled
    self.extract_data()
    
  def __getstate__(self): #stops pickle trying to save these fields
    d = dict(self.__dict__)
    [d.pop(k, None) for k in ['_scope_all_stats','_op_all_stats']]
    return d
  
  def __setstate__(self, d):
    self.__dict__.update(d)
    self.extract_data()
    
  
  def print_comparison(self, base, statsname, attrname):
    try:
      if hasattr(self, '_active_children'):
        this_total = 0
        base_total = 0
        base_children = getattr( getattr(base, statsname), 'children')

        for child in sorted(self._active_children, key=lambda child: child.name):
          cn = LayerName(child.name, flag='net_layer')
          value = getattr(child, attrname)
          this_total += value
          base_value = -1
          for i, bc in enumerate(base_children):
            try:
              bcn = LayerName(bc.name, flag='net_layer')
            except ValueError as errno:
              continue
            if bcn==cn:
              base_value = getattr(bc, attrname)
              base_total += base_value
              base_children.pop(i)
              break
          print('%d\t%d\t%d\t%s'%(base_value-value, value,base_value,cn))
        print('Totals: %d\t%d'%(this_total,base_total))
        print('\nUnmatched base nodes:')
        for i, bc in enumerate(base_children):
          base_value = getattr(bc, attrname)
          print('%d\t%s'%(base_value,bc.name))
        
        
      else:    
        this_total = getattr( getattr(self, statsname), attrname)
        
      base_total = getattr( getattr(base, statsname), attrname)
      return this_total - base_total, (this_total - base_total)/base_total
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
    
  
  def count_and_frac_delta(self, base, statsname, attrname):
#     self.print_comparison(base, statsname, attrname)
    try:
      if hasattr(self, '_active_children'):
        this_total = 0
        for child in self._active_children:
          this_total += getattr(child, attrname)
      else:    
        this_total = getattr( getattr(self, statsname), 'total_'+attrname)
        
      base_total = getattr( getattr(base, statsname), 'total_'+attrname)
      return this_total - base_total, (this_total - base_total)/base_total
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
        
  def serialize_data(self, graph, run_metadata_list):
    devnull = open(os.devnull, 'w')
    f = io.BytesIO()
    with stdout_redirector(f): #remove 'Parsing Inputs...'
      with RedirectStdStreams(stderr=devnull): #this stops a meaningless error on stderr
        profiler = Profiler(graph)

    for i, run_metadata in enumerate(run_metadata_list):
      profiler.add_step(i+1, run_metadata)
    
    #use these to print to stdout
#     profiler.profile_name_scope(ALL_OPTIONS_PRINT)

    all_opts = _build_options(ALL_OPTIONS)
    perf_opts = _build_options(PERF_OPTIONS)
    
    with stderr_redirector(f): #this stops some meaningless errors on stderr
      self._scope_all_stats_str = print_mdl.Profile('scope'.encode('utf-8'), all_opts.SerializeToString())
      self._op_all_stats_str = print_mdl.Profile('op'.encode('utf-8'), all_opts.SerializeToString())
      self._op_perf_stats_str = print_mdl.Profile('op'.encode('utf-8'), perf_opts.SerializeToString())
    #if you want to see what came down stderr run this:
    #print(f.getvalue().decode('utf-8'))


  def compare_nodes(self, base, attrname):
    
    def print_list(nodes, label, base_list=None):
      print('\n'+label)
      total = 0
      for w_node in sorted(nodes):
        total += w_node._value
        if base_list:
          base_idx = base_list.index(w_node)
          base_value = base_list[base_idx]._value
#           if abs(base_value - w_node._value) > 50000:
          if base_value != w_node._value:
            print('*** %d\t%d\t%s'%(w_node._value,base_value,w_node._key))
#           else:
#             print('%d\t%d\t%s'%(w_node._value,base_value,w_node._key))
        else:
          print('%d\t%s'%(w_node._value,w_node._key))
      print('Total = %d'%(total))
        
    if hasattr(self, '_op_all_stats'):
      base_rel_nodes, base_total = get_relevant_nodes(base._op_all_stats, attrname, 
                                                      print_nodes=False)
      self.plot_ops( base_rel_nodes, attrname )
      

    if hasattr(self, '_scope_all_stats'):
      this_rel_nodes, this_total = get_relevant_nodes(self._scope_all_stats, attrname)
      print('compressed total %s\t%d'%(attrname,this_total))
      base_rel_nodes, base_total = get_relevant_nodes(base._scope_all_stats, attrname, 
                                                      print_nodes=False)
      print('base total %s\t%d'%(attrname,base_total))
      self.plot_single_layers( base_rel_nodes, attrname )
      print('delta total %s\t%d'%(attrname,this_total-base_total))
      
      this_only_nodes = list(set(this_rel_nodes) - set(base_rel_nodes))
      both_nodes = list(set(this_rel_nodes) -  set(this_only_nodes))
      base_only_nodes = list(set(base_rel_nodes) - set(this_rel_nodes))
      
#       print_list(this_rel_nodes, attrname+' nodes in this model:')
      print_list(base_rel_nodes, attrname+' nodes in base model:')
      print_list(both_nodes, attrname+' nodes in both models:', base_rel_nodes)
      print_list(this_only_nodes, attrname+' nodes only in compressed model:')
      print_list(base_only_nodes, attrname+' nodes only in base model:')


  def extract_data(self):
#     def parse(stats_str):
#       stats = tfprof_output_pb2.GraphNodeProto()
#       stats.ParseFromString( stats_str )
#       return stats

    if hasattr(self, '_scope_all_stats_str'):
      self._scope_all_stats = tfprof_output_pb2.GraphNodeProto()
      self._scope_all_stats.ParseFromString( self._scope_all_stats_str )
      
    if hasattr(self, '_op_all_stats_str'):
      self._op_all_stats = tfprof_output_pb2.MultiGraphNodeProto()
      self._op_all_stats.ParseFromString( self._op_all_stats_str )

#     res = list(self._param_stats.DESCRIPTOR.fields_by_name.keys())
#     fields = ['name', 'tensor_value', 'run_count', 'exec_micros', 'accelerator_exec_micros',
#                 'cpu_exec_micros', 'requested_bytes', 'peak_bytes', 'residual_bytes', 
#                 'output_bytes', 'parameters', 'float_ops', 'devices', 'total_definition_count',
#                 'total_run_count', 'total_exec_micros', 'total_accelerator_exec_micros', 
#                 'total_cpu_exec_micros', 'total_requested_bytes', 'total_peak_bytes', 
#                 'total_residual_bytes', 'total_output_bytes', 'total_parameters', 
#                 'total_float_ops', 'shapes', 'input_shapes', 'children']

#     for k, v in stats.ListFields():
#       print(k.camelcase_name + ': ' + str(v))
#   #     value = stats.name
#   #     value = stats.total_parameters
#   #     value = stats.float_ops

#     for child in self._param_stats.children:
#       print('\n')
#       for k, v in child.ListFields():
#         print(k.camelcase_name + ': ' + str(v))


  def filter_active_path(self, stats, print_results=True):
    self._active_children = []
    self._inactive_children = []
    for child in sorted(stats.children, key=lambda child: child.name):
#       print(child.name)
      try:
        profile_layer = LayerName(child.name, 'net_layer_weights')
      except ValueError as errno:
        print("{0}: ValueError error({1})".format(child.name, errno))
        continue #is a non-layer e.g. Placeholder variable
      
      child_is_active = False
      if profile_layer.is_compressed():
        profile_layer_uncomp = profile_layer.uncompressed_version()
        if profile_layer_uncomp in self._net_desc:
          profile_K = profile_layer.K()
          if profile_K == self._net_desc[profile_layer_uncomp][0]:
            child_is_active = True  #this is a compressed layer in the active path
      else:      
        profile_layer_no_bn = profile_layer.remove_batch_norm().remove_biases()
        if profile_layer_no_bn not in self._net_desc:
          child_is_active = True #this is an uncompressed layer in the active path  
      
      if child_is_active:
        self._active_children.append(child)
      else:   
        self._inactive_children.append(child)
            
    if print_results:
      print('\n%d Active layers:'%(len(self._active_children)))
      for child in sorted(self._active_children, key=lambda child: child.name):
        print(child.name)         
      print('\n%d Inactive layers:'%(len(self._inactive_children)))
      for child in sorted(self._inactive_children, key=lambda child: child.name):
        print(child.name)         


  def aggregate_sub_nodes(self, rel_nodes):
    #aggregates together nodes to the 'layers' which they apply to allowing resource stats
    #to be assigned correctly to the layers of the model
    layer_dict = defaultdict( list )
#     terms_to_ignore = ['/weights','/biases','/read','/add','/Relu','/sub','/y',
#                        '/x2','/transpose','/Pad','/x1','/ToFloat','/strided_slice_1','/BiasAdd',
#                        '/MatMul','/sub','/concat','/values_1','/ToInt32','/transpose_1',
#                        '/batch_id','/truediv','/crops','/begin','/axis','/y2','/y1','/stack_1',
#                        '/values_2','/mul_1','/convolution','/perm','/Shape','/MaxPool',
#                        '/strided_slice','/stack','/Squeeze','/mul','/crop_size','/Ceil','/Ceil_1',
#                        '/generate_anchors','/input_3','/input_4','','']
    
    for i, node in enumerate(sorted(rel_nodes)):
      key = node._key
      idx = key.find('/')
      if idx != -1:
        key = key[idx+1:]
      
#       for term in terms_to_ignore:
#         if term in key:
#           key = key.replace(term,'')
      
      matched = False
      for i in range(1,4):
        idx = key.find('conv'+str(i))
        if idx != -1:
          layer = key[:idx+5]
          layer_dict[layer].append(node)
          matched = True
          break
      if matched:
        continue
        
#       idx = key.find('shortcut')
#       if idx != -1:
#         layer = key[:idx+8]
#         layer_dict[layer].append(node)
#         continue
          
      for ol in ordered_layers:
        if ol in key:
          layer_dict[ol].append(node)
          matched = True
          break
      if matched:
        continue
        
      layer_dict[key].append(node)
    
    agg_nodes = []
    print('\nAggregated nodes:')
    sum_unassigned = 0
    sum_assigned = 0
    for layer, nodes in sorted(layer_dict.items()):
      sum_ = 0
      keys_values = []
      for node in nodes:
        sum_ += node._value
        keys_values.append( (node._key, node._value) )
      agg_nodes.append(NodeWrapper(None, sum_, layer))
       
      if layer not in ordered_layers:
        clr = colour.GREEN
        sum_unassigned += sum_
      else:
        clr = colour.RED
        sum_assigned += sum_
      print(clr + str(sum_) + '\t' + layer + colour.END)
      for key, value in keys_values:
        print('\t' + key + '\t' + str(value))
    
    print('\n unassigned=%f%%, sum_assigned=%d, sum_unassigned=%d'%
          (100*sum_unassigned/(sum_assigned+sum_unassigned),sum_assigned,sum_unassigned))
    
    return agg_nodes
    
  def plot_single_layers(self, rel_nodes, metric_label ):
    xs = []
    ys = []
    layer_names = []
    block_start_idxs = {}
    block_end_idxs = {}  

    rel_nodes = self.aggregate_sub_nodes(rel_nodes)

    for i, node in enumerate(sorted(rel_nodes, key=lambda node_wrapper: sort_func(node_wrapper._key))):
      if node._key not in ordered_layers:
        continue #skip these few we can't assign to a layer
      
      for b in range(1,5):
        if 'block'+str(b) in node._key:
          block_end_idxs['block '+str(b)] = i+1
          if 'block '+str(b) not in block_start_idxs:
            block_start_idxs['block '+str(b)] = i+1
      
      disp_name = node._key.replace('bottleneck_v1/','').replace('/unit_','unit ')
      disp_name = disp_name.replace('bottleneck_v1','add')
      
#       idx = disp_name.find('/')
#       if idx != -1:
#         disp_name = disp_name[idx+1:]
      disp_name = re.sub(r'block[0-9]','', disp_name)
      disp_name = disp_name.replace('/',' / ')

      layer_names.append(disp_name)
      xs.append( i+1 )        
      ys.append( node._value )

      
    fig, ax = plt.subplots(1,1,figsize=(20,3))
    
    #do the block rectangle labels
    ymax = max(ys)
    height = ymax/7
    ymin = -height - ymax/20
    rectangles = {}
    for b in range(1,5):
      name = 'block '+str(b)
      start = block_start_idxs[name]
      end = block_end_idxs[name]
      width = end-start
      rectangles[name] = Rectangle((start, ymin), width, height, facecolor='green', alpha=0.5)
    
    start = block_end_idxs['block 3'] + 1
    end = block_start_idxs['block 4'] - 1
    width = end-start
    rectangles['RPN'] = Rectangle((start, ymin), width, height, facecolor='yellow', alpha=0.5)
      
    for r in rectangles:
        ax.add_artist(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width()/2.0
        cy = ry + rectangles[r].get_height()/2.0
        ax.annotate(r, (cx, cy), color='k', weight='bold', fontsize=14, ha='center', va='center')
    
    ax.plot(xs, ys,'o-')
    ax.set_ylim(ymin=ymin)
    ax.set_xlim(xmin=0, xmax=len(xs)+1)
        
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
#     plt.xticks(xs, layer_names, rotation=45)
    plt.xticks(xs, layer_names, rotation='vertical')
    plt.ylabel(metric_label.replace('_',' '), fontsize=16)
    plt.xlabel('layer', fontsize=16)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.45)
    plt.show()  
     
  def plot_ops(self, rel_nodes, metric_label):
    op_names = []
    metric_values = []
    bar_labels = []
    
    sum_ = 0
    for node in rel_nodes:
      sum_ += node._value

    for node in sorted(rel_nodes, key=lambda node: node._value):
#       print(node._key + '\t' + str(node._value))
      perc = 100*node._value/sum_
      if perc < 0.5:
        perc_str = '<0.5'
      else:
        perc_str = '%.0f'%(perc)
      bar_labels.append(perc_str+'%')
      op_names.append( node._key)        
      metric_values.append( node._value )
    ypos = np.arange(len(op_names))
      
    fig, ax = plt.subplots(1,1,figsize=(5,3))
    
    ax.barh(ypos, metric_values, align='center', color='green', ecolor='black')
    
    for i, (v,label) in enumerate(zip(metric_values,bar_labels)):
      ax.text(v + sum_/100, i, label, color='blue', verticalalignment='center')
    
    ax.set_yticks(ypos)
    ax.set_yticklabels(op_names)
    ax.set_xlabel(metric_label.replace('_',' '), fontsize=16)
    
    ax.set_ylim([-1,len(ypos)])
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.subplots_adjust(left=0.15, right=0.96, top=0.98, bottom=0.2)
    plt.show()  
     

  