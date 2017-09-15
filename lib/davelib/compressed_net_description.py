'''
Created on 13 Jul 2017

@author: david
'''
import tensorflow as tf
import numpy as np
import pickle as pi
import matplotlib.pyplot as plt
import sys  
import davelib.utils as utils

from functools import total_ordering
from mpl_toolkits.mplot3d import *
from davelib.layer_name import * 
from davelib.utils import colour

UNCOMPRESSED = 0 #enum used to indicate K=0 => uncompressed

def calc_Kmax(layer):
  shape = None
  for v in tf.global_variables():
    if '/'+layer.layer_weights() in v.name:
      shape = v.get_shape().as_list()
      break
  if not shape:
    raise ValueError('layer not found') 
      
  if len(shape)==4: #convolutional layer
    H,W,C,N = tuple(shape)
    Kmax = int(C*W*H*N / (C*W + H*N)) # if K > Kmax will have more parameters in sep layer
  elif len(shape)==2: #fully connected layer
    C,N = tuple(shape)
    Kmax = int(C*N / (C + N)) # if K > Kmax will have more parameters in sep layer
#     print('%s %d %d %d'%(layer, C, N, Kmax))
  return Kmax

def calc_all_Kmaxs():
  d = {}
  for layer in get_all_compressible_layers():
    Kmax = calc_Kmax(layer)
    d[layer] = Kmax
  return d
  
  
_all_Kmaxs_dict=None

def all_Kmaxs_dict():
  global _all_Kmaxs_dict
  if not _all_Kmaxs_dict:
    _all_Kmaxs_dict = calc_all_Kmaxs()
  return _all_Kmaxs_dict

def get_Ks(layer, K_fractions):
  Kmax = all_Kmaxs_dict()[layer]
  Ks = []
  for K_frac in K_fractions:
    K = int(K_frac * Kmax)
    if K == 0:
      K = 1
    elif K > Kmax:
      K = Kmax
    elif K < 0:
      continue #don't add a K it means this is uncompressed
    Ks.append(K)
  return Ks


def build_net_desc(Kfrac, compressed_layers):
  K_by_layer_dict = {}
  for layer_name in compressed_layers:
    Ks = get_Ks(layer_name, Kfrac)
    if len(Ks) > 0:
      K_by_layer_dict[layer_name] = Ks[0]
  net_desc = CompressedNetDescription(K_by_layer_dict)
  return net_desc

def build_pluri_net_desc(Kfracs, compressed_layers):
  Ks_by_layer_dict = {}
  for layer_name in compressed_layers:
    Ks = get_Ks(layer_name, Kfracs)
    if len(Ks) > 0:
      Ks_by_layer_dict[layer_name] = Ks
  net_desc = PluriNetDescription(Ks_by_layer_dict)
  return net_desc


class CompressionStep():
  def __init__(self, layer, K_old, K_new):
    self._layer = layer
    self._K_old = K_old
    self._K_new = K_new
    
  def __str__(self):
    return colour.RED + '%s: K:%d\u2192%d'%(self._layer,self._K_old,self._K_new) + colour.END

  

@total_ordering
class CompressedNetDescription(dict):
   
  def __init__(self, K_by_layer_dict):
    items = []
    for layer, K in K_by_layer_dict.items():
      if not type(K) is int:
        raise ValueError('Only accepts single K / layer')
      self[layer] = K
      items.extend( (layer, K) )
    self._key = tuple(items)

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
      return list(self.keys()) < list(other.keys())
  
  def __str__(self):
    return str(sorted(self.items()))
#   def __str__(self):
#     return '\n'.join(map(str, sorted(self.items())))

  def get_Kfrac(self):
    if len(self) == 0: # uncompressed
      return 0.0
    for layer, K in self.items():
      if 'block4' in layer and 'conv2' in layer:
        Kfrac = K / 768.0
        break
      elif 'block3' in layer and 'conv2' in layer:
        Kfrac = K / 384.0
        break
      elif 'block' not in layer and 'conv1' in layer:
        Kfrac = K / 20.0
        break
    return Kfrac

  def apply_compression_step(self, compression_step):
    layer, K_old, K_new = compression_step._layer,compression_step._K_old, compression_step._K_new
    
    if layer in self:
      true_K_old = self[layer]
    else:
      true_K_old = UNCOMPRESSED
    if true_K_old != K_old:
      raise ValueError(layer+': K_old=%d incorrect, true K_old=%d'%(K_old, true_K_old))
    
    K_by_layer_dict = {layer: K for layer, K in self.items() } #copy
    if K_new == UNCOMPRESSED:
      del K_by_layer_dict[layer]
    else:
      K_by_layer_dict[layer] = K_new
    return CompressedNetDescription(K_by_layer_dict)
  
  def get_differences(self, other):
    changes = []
    common_layers = list(set(self.keys()) & set(other.keys()))
    for layer in common_layers:
      K_self = self[layer]
      K_other = other[layer]
      if K_self != K_other:
        changes.append( CompressionStep(layer, K_other, K_self) )

    other_only_layers = list(set(other.keys()) - set(self.keys())) 
    for layer in other_only_layers:
      changes.append( CompressionStep(layer, other[layer], UNCOMPRESSED) )
      
    self_only_layers = list(set(self.keys()) - set(other.keys())) 
    for layer in self_only_layers:
      changes.append( CompressionStep(layer, UNCOMPRESSED, self[layer]) )
    return changes
  
  def K(self, layer):
    if layer in self:
      return self[layer]
    else:
      return UNCOMPRESSED


@total_ordering
class PluriNetDescription(dict):
   
  def __init__(self, Ks_by_layer_dict):
    items = []
    for layer, Ks in Ks_by_layer_dict.items():
      if not type(Ks) is list:
        raise ValueError('Only accepts list of Ks / layer')
      self[layer] = Ks
      items.extend( (layer, tuple(Ks)) )
    self._key = tuple(items)

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
      return list(self.keys())[0] < list(other.keys())[0]
  
#   def __str__(self):
#     return '\n'.join(map(str, sorted(self.items())))

  def get_Kfrac(self):
    raise ValueError('Not implemented anymore')
#     if len(self) == 0: # uncompressed
#       return 0.0
#     for layer, K in self.items():
#       if 'block4' in layer and 'conv2' in layer:
#         Kfrac = K[0] / 768.0
#         break
#       elif 'block3' in layer and 'conv2' in layer:
#         Kfrac = K[0] / 384.0
#         break
#       elif 'block' not in layer and 'conv1' in layer:
#         Kfrac = K[0] / 20.0
#         break
#     return Kfrac

  def apply_compression_step(self, compression_step):
    layer, K_old, K_new = compression_step._layer,compression_step._K_old, compression_step._K_new
    
    if layer in self:
      true_K_old = self[layer][0]
    else:
      true_K_old = utils.UNCOMPRESSED
    if true_K_old != K_old:
      raise ValueError('K_old=%d incorrect, true K_old=%s'%(K_old, true_K_old))
    
    self[layer] = K_new
