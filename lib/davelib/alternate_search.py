'''
Created on 13 Sep 2017

@author: david
'''
import davelib.utils as utils

from davelib.compressed_net_description import * 
from davelib.layer_name import LayerName, sort_func, ordered_layers
from copy import deepcopy
 
  
class AlternateSearch():
  
  def __init__(self, net_desc, efficiency_dict, performance_dict, perf_metric_increases_with_degradation,
               compressed_layers):
    self._net_desc = net_desc
    self._best_model = net_desc
    self._compression_step = None
    self._efficiency_dict = efficiency_dict
    self._performance_dict = performance_dict
    self._perf_metric_increases_with_degradation = perf_metric_increases_with_degradation
    self._ordered_layers = sorted(compressed_layers, key=lambda layer: sort_func(layer))
#     self._ordered_steps = self.build_search_order(simple=True)
    self._compressing = True
    self._results_dict = {}
    self._models_list = []
    self._this_cycle_start_idx = 0


  def _get_next_K(self, layer):
    sorted_keys = list(reversed(sorted(self._efficiency_dict[layer].keys())))
    K_old = self._net_desc[layer]
    
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
        K_new = utils.UNCOMPRESSED
        
    return K_new
    
    
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
  
  def _cycle_completed(self):
    #look back over the completed cycle and select the 'best' model
    opt_objectives = {}
    for net_desc in self._models_list[self._this_cycle_start_idx:]:
      new_efficiency_metric, new_performance_metric = self._results_dict[net_desc]
      objective = new_efficiency_metric #swap in alternative objectives here
      opt_objectives[objective] = net_desc 
    
    min_objective = list(sorted(opt_objectives))[0]
    self._best_model = opt_objectives[min_objective]
     
    if self._compressing:
      self._compressing = False
    else:
      self._compressing = True
      

  def get_next_model(self):
    next_layer, cycle_complete = self._get_next_layer()
    
    if cycle_complete: #end of this set of un/compressions
      self._cycle_completed()
      
    K_new = self._get_next_K(next_layer)
    K_old = self._net_desc[next_layer]
    if K_new == K_old:
      return self.get_next_model() #this layer is fully compressed so try the next layer
    
    comp_step = CompressionStep(next_layer, K_old, K_new)
    
    self._net_desc = deepcopy(self._best_model)
    self._net_desc.apply_compression_step(comp_step)
    if self._net_desc in self._results_dict: 
      return self.get_next_model() #already run this model
      
    self._models_list.append(self._net_desc)
    return self._net_desc, comp_step  
    
  def set_model_results(self, net_desc, new_efficiency_metric, new_performance_metric):
    if self._models_list[-1] != net_desc:
      raise ValueError('model results don\'t match present model')
    self._results_dict[self._net_desc] = (new_efficiency_metric, new_performance_metric)
    
    
    
    
    
    
        
