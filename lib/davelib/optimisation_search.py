'''
Created on 13 Sep 2017

@author: david
'''

  
class BruteForceSearch():
  
  def __init__(self, efficiency_dict, performance_dict, perf_metric_increases_with_degradation,
               compressed_layers):
    self._layer_K_dict = {}
    self._efficiency_dict = efficiency_dict
    self._performance_dict = performance_dict
    self._perf_metric_increases_with_degradation = perf_metric_increases_with_degradation
    
    self._ordered_layers = compressed_layers
    self._ordered_steps = self.build_search_order(simple=True)

    
  def _get_K_old_new(self, layer, sorted_keys):
    def _get_next_K(K_old):
      idx = sorted_keys.index( K_old )
      if idx+1 < len(sorted_keys):
        K_new = sorted_keys[idx + 1]
      else:
        K_new = 0
      return K_new
      
    sorted_keys = list(reversed(sorted(self._efficiency_dict[layer].keys())))
    if layer in self._layer_K_dict:
      K_old = self._layer_K_dict[layer][0]
      K_new = _get_next_K(K_old, sorted_keys)
    else:
      K_old = 0
      K_new = sorted_keys[0] #start with the largest K value
      efficiency_delta = self._efficiency_dict[layer][K_new]
      if efficiency_delta == 0:
        K_new = _get_next_K(K_new) #hack to skip the first flops item
    return K_old, K_new

    
  def build_search_order(self, simple):
    grad_dict = {}
    perf_delta_dict = {}
    efficiency_delta_dict = {}
  
    for layer in self._efficiency_dict:
      K_old, K_new = self._get_K_old_new(layer)  
      if K_new == 0: #max compression reached for this layer
        continue
      elif K_old != 0: #already compressed this layer a bit
        efficiency_delta = self._efficiency_dict[layer][K_new] - self._efficiency_dict[layer][K_old]
        performance_delta = self._performance_dict[layer][K_new] - self._performance_dict[layer][K_old]
      else: #this layer hasn't been compressed at all yet
        efficiency_delta = self._efficiency_dict[layer][K_new]
        performance_delta = self._performance_dict[layer][K_new]
        
      if efficiency_delta==0:
        continue #ignore steps that don't change efficiency
        
      if simple:
        efficiency_grad = efficiency_delta
      else:
        efficiency_grad = efficiency_delta / performance_delta
      if not self._perf_metric_increases_with_degradation:
        efficiency_grad *= -1.0
      grad_dict[efficiency_grad] = CompressionStep(layer,K_old, K_new) 
      perf_delta_dict[layer] = performance_delta
      efficiency_delta_dict[layer] = efficiency_delta
      
    return sorted(grad_dict).values()
#     step_with_min_grad = grad_dict[grad_min]
# 
#     layer_K_dict[layer_with_min_grad] = [K_new]
#     expected_perf_delta = perf_delta_dict[layer_with_min_grad]
#     expected_efficiency_delta = efficiency_delta_dict[layer_with_min_grad]

    
  def _get_compression(self, layer):
    def get_next_K(K_old):
      idx = sorted_keys.index( K_old )
      if idx+1 < len(sorted_keys):
        K_new = sorted_keys[idx + 1]
      else:
        K_new = 0
      return K_new
      
    sorted_keys = list(reversed(sorted(self._efficiency_dict[layer].keys())))
    if layer in self._layer_K_dict:
      K_old = self._layer_K_dict[layer][0]
      K_new = get_next_K(K_old)
    else:
      K_old = 0
      K_new = sorted_keys[0] #start with the largest K value
      efficiency_delta = self._efficiency_dict[layer][K_new]
      if efficiency_delta == 0:
        K_new = get_next_K(K_new) #hack to skip the first flops item
        
    return CompressionStep(layer, K_old, K_new)


  def get_next_model(self):
    
    if not self._this_compression: #first step
      self._this_compression = self._ordered_steps[0]
      return self._this_compression
    
    
    idx = self._ordered_steps.index( self._this_compression )
    if idx+1 < len(self._ordered_steps):
      next_step = self._ordered_steps[idx+1]
    
      
    


def _get_next_model_brute_force(self, efficiency_dict, performance_dict, 
                      perf_metric_increases_with_degradation, simple=True):
    def get_K_old_new(layer):
      def get_next_K(K_old):
        idx = sorted_keys.index( K_old )
        if idx+1 < len(sorted_keys):
          K_new = sorted_keys[idx + 1]
        else:
          K_new = 0
        return K_new
        
      sorted_keys = list(reversed(sorted(efficiency_dict[layer].keys())))
      if layer in layer_K_dict:
        K_old = layer_K_dict[layer][0]
        K_new = get_next_K(K_old)
      else:
        K_old = 0
        K_new = sorted_keys[0] #start with the largest K value
        efficiency_delta = efficiency_dict[layer][K_new]
        if efficiency_delta == 0:
          K_new = get_next_K(K_new) #hack to skip the first flops item
      return K_old, K_new

    grad_dict = {}
    perf_delta_dict = {}
    efficiency_delta_dict = {}
    for layer in efficiency_dict:
      K_old, K_new = get_K_old_new(layer)  
      if K_new == 0: #max compression reached for this layer
        continue
      elif K_old != 0: #already compressed this layer a bit
        efficiency_delta = efficiency_dict[layer][K_new] - efficiency_dict[layer][K_old]
        performance_delta = performance_dict[layer][K_new] - performance_dict[layer][K_old]
      else: #this layer hasn't been compressed at all yet
        efficiency_delta = efficiency_dict[layer][K_new]
        performance_delta = performance_dict[layer][K_new]
        
      if simple:
        efficiency_grad = efficiency_delta
      else:
        efficiency_grad = efficiency_delta / performance_delta
      if not perf_metric_increases_with_degradation:
        efficiency_grad *= -1.0
      grad_dict[efficiency_grad] = layer 
      perf_delta_dict[layer] = performance_delta
      efficiency_delta_dict[layer] = efficiency_delta
      
    grad_min = list(sorted(grad_dict))[0]
    layer_with_min_grad = grad_dict[grad_min]
    K_old, K_new = get_K_old_new(layer_with_min_grad)
    layer_K_dict[layer_with_min_grad] = [K_new]
    expected_perf_delta = perf_delta_dict[layer_with_min_grad]
    expected_efficiency_delta = efficiency_delta_dict[layer_with_min_grad]

    print(colour.RED + '%s: K:%d\u2192%d'%(layer_with_min_grad,K_old,K_new) + colour.END)

    return layer_K_dict, expected_perf_delta, expected_efficiency_delta, \
            NetChange(layer_with_min_grad, K_old, K_new)
