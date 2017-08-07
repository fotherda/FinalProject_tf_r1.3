'''
Created on 10 Jul 2017

@author: david
'''
from functools import total_ordering
from collections import OrderedDict

@total_ordering
class LayerName(str):

  def __new__(cls, value, flag=None):
    if flag in ['net_layer','net_layer_weights']:
      idx = value.index('/')
      value = value[idx+1:]
    value = value.replace('/weights','')
    # explicitly only pass value to the str constructor
    return super(LayerName, cls).__new__(cls, value)
  
  def __init__(self, value, flag=None):
        # ... and don't even call the str initializer 
      self.flag = flag
      if 'weights' in value:
        self._has_weights = True
      else:
        self._has_weights = False
        
  def __lt__(self, other): #order layers as they come in the actual network
    layers = get_all_compressible_layers()
    return layers.index(self) < layers.index(other)

  def net_layer_weights(self, net):
    return net + '/' + self.layer_weights()
      
  def net_layer(self, net):
    return net + '/' + self
      
  def layer_weights(self):
    if self._has_weights:
      return self + '/weights'
    else:
      return self 
    
  def sep_version(self):
    return LayerName(self.layer_weights().replace('conv', 'convsep'), self.flag)
    
    
__COMPRESSIBLE_LAYERS__ = None

def get_all_compressible_layers():  
  global __COMPRESSIBLE_LAYERS__
  
  if __COMPRESSIBLE_LAYERS__:
    return __COMPRESSIBLE_LAYERS__
  
  __COMPRESSIBLE_LAYERS__ = []
  __COMPRESSIBLE_LAYERS__.append( LayerName('conv1/weights') )
  d = {'1':3, '2':4, '3':23, '4':3} #for resnet101. Change for other size resnets
  block_layer_dict = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

  for block_num, num_layers in block_layer_dict.items():
    for unit_num in range(1,num_layers+1):
#       for layer_type in ['conv2']:
      for layer_type in ['conv1','conv2','conv3']:
        __COMPRESSIBLE_LAYERS__.append( LayerName('block'+block_num+'/unit_'+str(unit_num)+
                               '/bottleneck_v1/' + layer_type + '/weights') )
      if unit_num == 1:
        __COMPRESSIBLE_LAYERS__.append( LayerName('block'+block_num+'/unit_'+str(unit_num)+
                               '/bottleneck_v1/shortcut/weights') )
    if block_num == '3':
      __COMPRESSIBLE_LAYERS__.append( LayerName('rpn_conv/3x3/weights') )
       
  __COMPRESSIBLE_LAYERS__.append( LayerName('cls_score/weights') ) 
  __COMPRESSIBLE_LAYERS__.append( LayerName('bbox_pred/weights') ) 
  return __COMPRESSIBLE_LAYERS__


def remove_bottleneck_1_3_shortcut_layers(layer_names):  
  filtered = []
  for v in layer_names:
    if 'block' not in v:
      filtered.append(v)
    elif 'conv2' in v:
      filtered.append(v)
  return filtered

def remove_bottleneck_not_unit1(layer_names):  
  filtered = []
  for v in layer_names:
    if 'unit' not in v:
      filtered.append(v)
    elif 'unit_1/' in v:
      filtered.append(v)
  return filtered

def remove_layers_after_block3(layer_names):  
  filtered = [ v for v in layer_names[:] if 'block4' not in v and 'rpn_conv' not in v ]
  return filtered

