'''
Created on 13 Sep 2017

@author: david
'''
import davelib.utils as utils
import pickle as pi

from davelib.compressed_net_description import * 
from davelib.layer_name import LayerName, sort_func, ordered_layers
from copy import deepcopy
from collections import OrderedDict
from enum import Enum
 
  
def change_missing_compressions(template_dict, net_desc):
  #if any K in the net_desc isn't in the template dict change it to the nearest one that is
  K_by_layer_dict = {}
  for layer, K_net in net_desc.items():
#   for layer, d in template_dict.items():
#     K_net = net_desc[layer]
    d = template_dict[layer]
    if not K_net in d:
      diffs = {}
      for K in d:
        diff = abs(K - K_net)
        diffs[diff] = K
      min_diff = min(diffs)
      closest_K = diffs[min_diff]
      K_by_layer_dict[layer] = closest_K
    else:
      K_by_layer_dict[layer] = K_net
  return CompressedNetDescription(K_by_layer_dict)
  
class ObjectiveType(Enum):
  EFFICIENCY = 1
  PERFORMANCE = 2
  EFFICIENCY_AND_PERFORMANCE = 3  
  
class AlternateSearch():
  
  def __init__(self, initial_net_desc, efficiency_dict, performance_dict, perf_metric_increases_with_degradation,
               compressed_layers, compress_only, objective_type):
    initial_net_desc = change_missing_compressions(efficiency_dict, initial_net_desc)
    self._best_model = initial_net_desc
    self._compression_step = None
    self._efficiency_dict = efficiency_dict
    self._performance_dict = performance_dict
    self._perf_metric_increases_with_degradation = perf_metric_increases_with_degradation
    self._ordered_layers = sorted(compressed_layers, key=lambda layer: sort_func(layer))
    self._compressing = True
    self._end_of_cycle_results = []
    self._models_dict = OrderedDict({})
    self._this_cycle_start_idx = 0
    self._compress_only = compress_only
    self._objective_type = objective_type

  def _get_next_K(self, layer):
    sorted_keys = list(reversed(sorted(self._efficiency_dict[layer].keys())))
    K_old = self._best_model.K(layer)
    if K_old == UNCOMPRESSED:
      idx = -1
    else:
      idx = sorted_keys.index( K_old )
    
    if self._compressing:
      if idx+1 < len(sorted_keys):
        K_new = sorted_keys[idx + 1]
      else:
        K_new = K_old
    else:
      if idx-1 >= 0:
        K_new = sorted_keys[idx - 1]
      else:
        K_new = UNCOMPRESSED
    return K_old, K_new
    
  def _get_next_layer(self):
    cycle_complete = False
    if not self._compression_step: #first step
      next_layer = self._ordered_layers[0]
    else:
      idx = self._ordered_layers.index(self._compression_step._layer)
      if idx+1 < len(self._ordered_layers):
        next_layer = self._ordered_layers[idx+1]
      else:
        next_layer = self._ordered_layers[0]
        cycle_complete = True
    return next_layer, cycle_complete
  
  
  def _best_effic(self):
    opt_objectives = {}
    for net_desc, results in list(self._models_dict.items())[self._this_cycle_start_idx:]:
      objective = results._new_efficiency_metric 
      opt_objectives[net_desc] = objective
    best_net_desc = list(sorted(opt_objectives, key=opt_objectives.get))[0]
    return best_net_desc, opt_objectives[best_net_desc]

  def _best_perf(self):
    opt_objectives = {}
    for net_desc, results in list(self._models_dict.items())[self._this_cycle_start_idx:]:
      objective = results._new_performance_metric 
      opt_objectives[net_desc] = objective
    best_net_desc = list(sorted(opt_objectives, key=opt_objectives.get))[0]
    return best_net_desc, opt_objectives[best_net_desc]
      
      
  def _best_perf_for_best_effic(self):
    #look back over the completed cycle and select the 'best' model
    best_net_desc, best_effic = self._best_effic()
    opt_objectives = {}
    for net_desc, results in list(self._models_dict.items())[self._this_cycle_start_idx:]:
      if results._new_efficiency_metric == best_effic:
        opt_objectives[net_desc] = results._actual_perf_delta
        
    if self._perf_metric_increases_with_degradation:
      best_net_desc = list(sorted(opt_objectives, key=opt_objectives.get))[0]
    else:
      best_net_desc = list(sorted(opt_objectives, key=opt_objectives.get))[-1]
      
    return best_net_desc, opt_objectives[best_net_desc]
    
    
  def apply_objective(self):
    if self._objective_type == ObjectiveType.EFFICIENCY:
      best_net_desc, objective_val = self._best_effic()
    elif self._objective_type == ObjectiveType.PERFORMANCE:
      best_net_desc, objective_val = self._best_perf()
    elif self._objective_type == ObjectiveType.EFFICIENCY_AND_PERFORMANCE:
      best_net_desc, objective_val = self._best_perf_for_best_effic()
    return best_net_desc, objective_val
    
  def _cycle_completed(self):
    if len(self._models_dict) == self._this_cycle_start_idx: #means no new models were tested this cycle
      print(colour.GREEN + self._comp_str() + 'no change')
    else:
      best_net_desc, objective_val = self.apply_objective
      changes = best_net_desc.get_differences(self._best_model)
      assert len(changes)==1, 'Should only be one change in net / cycle' 
      print(colour.GREEN + self._comp_str() + 'change: ' + str(changes[0]))
      print(str(best_net_desc) + '\n')
      self._best_model = best_net_desc
      
    res = self._models_dict[self._best_model]
    self._end_of_cycle_results.append(res)
    with open('alt_srch_res','wb') as f:
      pi.dump(self._end_of_cycle_results, f)

    if not self._compress_only:
      if self._compressing:
        self._compressing = False
      else:
        self._compressing = True
    self._this_cycle_start_idx = len(self._models_dict)

  def _comp_str(self):
    if self._compressing: return 'compressing cycle '
    else: return 'uncompressing cycle '
    
  def get_next_model(self):
    if len(self._end_of_cycle_results) >= 2:
      if self._end_of_cycle_results[-1] == self._end_of_cycle_results[-2]:
        return None, None #indicates we've converged
      
    while True:
      next_layer, cycle_complete = self._get_next_layer()
      
      if cycle_complete: #end of this set of un/compressions
        self._cycle_completed()
        
      K_old, K_new = self._get_next_K(next_layer)
      self._compression_step = CompressionStep(next_layer, K_old, K_new)
      if K_new == K_old:
        continue  #this layer is fully un/compressed so try the next layer
      
      next_model = self._best_model.apply_compression_step( self._compression_step )
      if next_model in self._models_dict: 
        continue #already run this model so try the next layer
      break  
        
    self._models_dict[next_model] = None #this will be rplaced with the results once it's run
    print(self._compression_step)
#     print(next_model)
    return next_model, self._compression_step  
    
  def set_model_results(self, this_iter_res):
    net_desc = this_iter_res._net_desc
    if next(reversed( self._models_dict )) != net_desc:
      raise ValueError('model results don\'t match present model')
    self._models_dict[net_desc] = this_iter_res
    
    
