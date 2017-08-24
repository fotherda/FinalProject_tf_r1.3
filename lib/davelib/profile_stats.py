
'''
Created on 28 Jul 2017

@author: david
'''
import sys, os, logging, io
import tensorflow as tf
import copy

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
    'account_displayed_op_only': False,
    'select': ['bytes','peak_bytes','residual_bytes','output_bytes','params','op_types','tensor_value','input_shapes','occurrence'],
    'viz': True,
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

PERF_OPTIONS_PRINT = copy.deepcopy(PERF_OPTIONS)
PERF_OPTIONS_PRINT['output'] = ''
PARAM_OPTIONS_PRINT = copy.deepcopy(PARAM_OPTIONS)
PARAM_OPTIONS_PRINT['output'] = ''

def total_bytes_count(profile_stats):
  return profile_stats._scope_all_stats.total_peak_bytes + \
         profile_stats._scope_all_stats.total_output_bytes   

class ProfileStats(object):
  '''
  holds profiling information for a network
  '''

  def __init__(self, run_metadata_list, graph):
    #these can be pickled
    self.serialize_data(graph, run_metadata_list)
    #these can't be pickled
    self.extract_data()
    
  def __getstate__(self): #stops pickle trying to save these fields
    d = dict(self.__dict__)
    [d.pop(k, None) for k in ['_param_stats','_perf_stats','_op_param_stats',
                              '_op_perf_stats','_scope_all_stats','_op_all_stats']]
    return d
  
  def __setstate__(self, d):
    self.__dict__.update(d)
    self.extract_data()
  
  def count_delta(self, other, statsname, attrname):
    try:
      other_total = getattr( getattr(other, statsname), attrname)
      this_total = getattr( getattr(self, statsname), attrname)
#       return  this_total - other_total
      return None
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise
    
  def frac_delta(self, other, statsname, attrname):
    try:
      other_total = getattr( getattr(other, statsname), attrname)
      this_total = getattr( getattr(self, statsname), attrname)
      return (this_total - other_total)/other_total
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise

  def total_bytes_count_delta(self, other):   
    try:
      return  total_bytes_count(self) - total_bytes_count(other)
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise
        
  def total_bytes_frac_delta(self, other):   
    try:
      return  (total_bytes_count(self) - total_bytes_count(other)) / total_bytes_count(other)
    except AttributeError as errno:
      print("AttributeError error({0})".format(errno))
      return None
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise
        
#   def total_bytes_count_delta(self, other):   
#     other_total = other._param_stats.total_requested_bytes + \
#                   other._perf_stats.total_requested_bytes
#     this_total = self._param_stats.total_requested_bytes + \
#                   self._perf_stats.total_requested_bytes
#     return  this_total - other_total
#         
#   def total_bytes_frac_delta(self, other):   
#     other_total = other._param_stats.total_requested_bytes + \
#                   other._perf_stats.total_requested_bytes
#     this_total = self._param_stats.total_requested_bytes + \
#                   self._perf_stats.total_requested_bytes
#     return (this_total - other_total)/other_total
#         
  def serialize_data(self, graph, run_metadata_list):
    devnull = open(os.devnull, 'w')
    f = io.BytesIO()
    with RedirectStdStreams(stderr=devnull): #this stops a meaningless error on stderr
#     with stdout_redirector(f): #remove 'Parsing Inputs...'
      profiler = Profiler(graph)

    for i, run_metadata in enumerate(run_metadata_list):
      profiler.add_step(i+1, run_metadata)
    
    #use these to print to stdout
#     profiler.profile_name_scope(PARAM_OPTIONS_PRINT)
#     profiler.profile_name_scope(PERF_OPTIONS_PRINT)

    param_opts = _build_options(PARAM_OPTIONS)
    perf_opts = _build_options(PERF_OPTIONS)
    all_opts = _build_options(ALL_OPTIONS)
    
    with stderr_redirector(f): #this stops some meaningless errors on stderr
      self._param_stats_str = print_mdl.Profile('scope'.encode('utf-8'), param_opts.SerializeToString())
      self._perf_stats_str = print_mdl.Profile('scope'.encode('utf-8'), perf_opts.SerializeToString())
      self._scope_all_stats_str = print_mdl.Profile('scope'.encode('utf-8'), all_opts.SerializeToString())
      self._op_param_stats_str = print_mdl.Profile('op'.encode('utf-8'), param_opts.SerializeToString())
      self._op_perf_stats_str = print_mdl.Profile('op'.encode('utf-8'), perf_opts.SerializeToString())
      self._op_all_stats_str = print_mdl.Profile('op'.encode('utf-8'), all_opts.SerializeToString())

    #if you want to see what came down stderr run this:
    #print(f.getvalue().decode('utf-8'))

  def extract_data(self):
    def parse(stats_str):
      stats = tfprof_output_pb2.GraphNodeProto()
      stats.ParseFromString( stats_str )
      return stats
      
    self._param_stats = parse( self._param_stats_str )
    self._perf_stats = parse( self._perf_stats_str )
    if hasattr(self, '_scope_all_stats_str'):
      self._scope_all_stats = parse( self._scope_all_stats_str )
    
    self._op_param_stats = parse( self._op_param_stats_str )
    self._op_perf_stats = parse( self._op_perf_stats_str )
    if hasattr(self, '_op_all_stats_str'):
      self._op_all_stats = parse( self._op_all_stats_str )

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
#   
#     for child in param_stats.children:
#       print('\n')
#       for k, v in child.ListFields():
#         print(k.camelcase_name + ': ' + str(v))

  
  