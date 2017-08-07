
'''
Created on 28 Jul 2017

@author: david
'''
import sys, os, logging, io

from collections import defaultdict
from tensorflow.python.profiler.model_analyzer import _build_options, Profiler
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.core.profiler import tfprof_output_pb2
from davelib.utils import show_all_variables, RedirectStdStreams, stderr_redirector, stdout_redirector


PARAM_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 1,
    'min_float_ops': 0,
    'device_regexes': ['.*'],
    'order_by': 'params',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['bytes','params','op_types','tensor_value'],
    'viz': False,
    'dump_to_file': '',
#     'output': ''
#     'output': 'timeline:outfile=timeline_param'
    'output': 'file:outfile=profile_param_stats'
}
PERF_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 0,
    'min_float_ops': 1,
    'device_regexes': ['.*'],
    'order_by': 'micros',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['float_ops','micros','bytes','op_types','tensor_value'],
    'viz': False,
    'dump_to_file': '',
#     'output': ''
#     'output': 'timeline:outfile=timeline_perf'
    'output': 'file:outfile=profile_perf_stats'
}

class ProfileStats(object):
  '''
  holds profiling information for a network
  '''

  def __init__(self, run_metadata_list, graph):
    #these can be pickled
    self._param_stats_str, self._perf_stats_str, self._op_param_stats_str, \
          self._op_perf_stats_str = self.serialize_data(graph, run_metadata_list)
    #these can't be pickled
    self._param_stats, self._perf_stats, self._op_param_stats, self._op_perf_stats = \
          self.extract_data()
    
  def __getstate__(self): #stops pickle trying to save these fields
    d = dict(self.__dict__)
    [d.pop(k, None) for k in ['_param_stats','_perf_stats','_op_param_stats','_op_perf_stats']]
    return d
  
  def __setstate__(self, d):
    self.__dict__.update(d)
    self._param_stats, self._perf_stats, self._op_param_stats, self._op_perf_stats = \
          self.extract_data()
  
  def count_delta(self, other, statsname, attrname):
    other_total = getattr( getattr(other, statsname), attrname)
    this_total = getattr( getattr(self, statsname), attrname)
    return  this_total - other_total
    
  def frac_delta(self, other, statsname, attrname):
    other_total = getattr( getattr(other, statsname), attrname)
    this_total = getattr( getattr(self, statsname), attrname)
    return (this_total - other_total)/other_total
        
  def serialize_data(self, graph, run_metadata_list):
    devnull = open(os.devnull, 'w')
    f = io.BytesIO()
    with RedirectStdStreams(stderr=devnull): #this stops a meaningless error on stderr
#     with stdout_redirector(f): #remove 'Parsing Inputs...'
      profiler = Profiler(graph)

    for i, run_metadata in enumerate(run_metadata_list):
      profiler.add_step(i+1, run_metadata)
    
    param_opts = _build_options(PARAM_OPTIONS)
    perf_opts = _build_options(PERF_OPTIONS)
    
    with stderr_redirector(f): #this stops some meaningless errors on stderr
      param_stats_str = print_mdl.Profile('scope'.encode('utf-8'), param_opts.SerializeToString())
      perf_stats_str = print_mdl.Profile('scope'.encode('utf-8'), perf_opts.SerializeToString())
      op_param_stats_str = print_mdl.Profile('op'.encode('utf-8'), param_opts.SerializeToString())
      op_perf_stats_str = print_mdl.Profile('op'.encode('utf-8'), perf_opts.SerializeToString())

    #if you want to see what came down stderr run this:
    #print(f.getvalue().decode('utf-8'))

    return param_stats_str, perf_stats_str, op_param_stats_str, op_perf_stats_str
  
  def extract_data(self):
  #   res = list(stats.DESCRIPTOR.fields_by_name.keys())
#     fields = ['name', 'tensor_value', 'exec_micros', 'requested_bytes', 'parameters', 
#      'float_ops', 'inputs', 'device', 'total_exec_micros', 'total_requested_bytes', 
#      'total_parameters', 'total_float_ops', 'total_inputs', 'shapes', 'children']
    
    param_stats = tfprof_output_pb2.GraphNodeProto()
    param_stats.ParseFromString( self._param_stats_str )
    perf_stats = tfprof_output_pb2.GraphNodeProto()
    perf_stats.ParseFromString( self._perf_stats_str )
    
    op_param_stats = tfprof_output_pb2.MultiGraphNodeProto()
    op_param_stats.ParseFromString( self._op_param_stats_str )
    op_perf_stats = tfprof_output_pb2.MultiGraphNodeProto()
    op_perf_stats.ParseFromString( self._op_perf_stats_str )
    
#     for k, v in stats.ListFields():
#       print(k.camelcase_name + ': ' + str(v))
#   #     value = stats.name
#   #     value = stats.total_parameters
#   #     value = stats.float_ops
#   
#     for child in param_stats.children:
#       print('\n')
#       for k, v in child.ListFields():
#         print(k.camelcase_name + ': ' + str(v))

    return param_stats, perf_stats, op_param_stats, op_perf_stats
  
  
  