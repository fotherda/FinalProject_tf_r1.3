'''
Created on 13 Sep 2017

@author: david
'''
import davelib.utils as utils
import pickle as pi

from davelib.compressed_net_description import * 
from davelib.layer_name import LayerName, sort_func, ordered_layers
from copy import deepcopy
 
  
def change_missing_compressions(template_dict, net_desc):
  #if any K in the net_desc isn't in the template dict change it to the nearest one that is
  K_by_layer_dict = {}
  for layer, d in template_dict.items():
    K_net = net_desc[layer]
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
  
class AlternateSearch():
  
  def __init__(self, initial_net_desc, efficiency_dict, performance_dict, perf_metric_increases_with_degradation,
               compressed_layers):
    initial_net_desc = change_missing_compressions(efficiency_dict, initial_net_desc)
    self._net_desc = initial_net_desc
    self._best_model = initial_net_desc
    self._compression_step = None
    self._efficiency_dict = efficiency_dict
    self._performance_dict = performance_dict
    self._perf_metric_increases_with_degradation = perf_metric_increases_with_degradation
    self._ordered_layers = sorted(compressed_layers, key=lambda layer: sort_func(layer))
    self._compressing = True
    self._results_dict = {}
    self._end_of_cycle_results = []
    self._models_list = []
    self._this_cycle_start_idx = 0

  def _get_next_K(self, layer):
    sorted_keys = list(reversed(sorted(self._efficiency_dict[layer].keys())))
    K_old = self._net_desc.K(layer)
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
      objective = self._results_dict[net_desc]._new_efficiency_metric #swap in alternative objectives here
      opt_objectives[net_desc] = objective
      
    if len(opt_objectives) == 0: #means no new models were tested this cycle
      print(colour.GREEN + self._comp_str() + 'no change')
    else:
      best_net_desc = list(sorted(opt_objectives, key=opt_objectives.get))[0]
      changes = best_net_desc.get_differences(self._best_model)
      assert len(changes)==1, 'Should only be one change in net / cycle' 
      print(colour.GREEN + self._comp_str() + 'change: ' + str(changes[0]))
      print(str(best_net_desc) + '\n')
      self._best_model = best_net_desc
      
    self._end_of_cycle_results.append(self._results_dict[self._best_model])
    pi.dump(self._end_of_cycle_results, open('alt_srch_res','wb'))

    if self._compressing:
      self._compressing = False
    else:
      self._compressing = True
    self._this_cycle_start_idx = len(self._models_list)

  def _comp_str(self):
    if self._compressing: return 'compressing cycle '
    else: return 'uncompressing cycle '
    
  def get_next_model(self):
    next_layer, cycle_complete = self._get_next_layer()
    
    if cycle_complete: #end of this set of un/compressions
      self._cycle_completed()
      
    K_new = self._get_next_K(next_layer)
    K_old = self._best_model.K(next_layer)
    self._compression_step = CompressionStep(next_layer, K_old, K_new)
    if K_new == K_old:
      return self.get_next_model() #this layer is fully compressed so try the next layer
    
    self._net_desc = self._best_model.apply_compression_step( self._compression_step )
    if self._net_desc in self._results_dict: 
      return self.get_next_model() #already run this model
      
    self._models_list.append(deepcopy(self._net_desc))
    print(self._compression_step)
#     print(self._net_desc)
    return self._net_desc, self._compression_step  
    
  def set_model_results(self, this_iter_res):
    net_desc = this_iter_res._net_desc
    if self._models_list[-1] != net_desc or self._net_desc != net_desc:
      raise ValueError('model results don\'t match present model')
    self._results_dict[self._net_desc] = this_iter_res
    
    
