'''
Created on 13 Jul 2017

@author: david
'''
import tensorflow as tf
import re, os
import numpy as np
import pickle as pi
import matplotlib.pyplot as plt
import sys  
import davelib.utils as utils

from functools import total_ordering
from mpl_toolkits.mplot3d import *
from davelib.layer_name import * 
from davelib.utils import colour
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter


UNCOMPRESSED = 0 #enum used to indicate K=0 => uncompressed

DECOMPRESSION = -1
COMPRESSION = 1
NO_CHANGE = 0

def calc_Kmax(layer):
  shape = None
  for v in tf.global_variables():
    if '/'+layer.layer_weights() in v.name:
      shape = v.get_shape().as_list()
      break
  if not shape:
    raise ValueError('%s - layer not found'%(layer)) 
      
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
    filename = 'all_Kmaxs_dict'
    if os.path.isfile(filename):
      _all_Kmaxs_dict = pi.load(open(filename,'rb'))
    else:
      _all_Kmaxs_dict = calc_all_Kmaxs()
      pi.dump(_all_Kmaxs_dict, open(filename,'wb'))
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

def get_Kfrac(layer, K):
  Kmax = all_Kmaxs_dict()[layer]
  if K == UNCOMPRESSED:
    Kfrac = 1.0
  else:
    Kfrac = K / Kmax
  return Kfrac


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

  def K_new(self):
    if self._K_new == UNCOMPRESSED:
      return sys.maxsize
    else:
      return self._K_new

  def K_old(self):
    if self._K_old == UNCOMPRESSED:
      return sys.maxsize
    else:
      return self._K_old

  def type(self):
    if self.K_new() < self.K_old():
      return COMPRESSION
    elif self.K_new() > self.K_old():
      return DECOMPRESSION
    else:
      return NO_CHANGE  

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
    
  def plot_compression_profile(self):
    xs = []
    ys = []
    layer_names = []
    block_start_idxs = {}
    block_end_idxs = {}  

    
    
    for i, layer in enumerate( get_all_compressible_layers() ):
#     for i, layer in enumerate(sorted(self, key=lambda layer: sort_func(layer))):
      if layer in self:
        K = self[layer]
      else:
        K = UNCOMPRESSED
      for b in range(1,5):
        if 'block'+str(b) in layer:
          block_end_idxs['block '+str(b)] = i+1
          if 'block '+str(b) not in block_start_idxs:
            block_start_idxs['block '+str(b)] = i+1
      
      disp_name = layer.replace('bottleneck_v1/','').replace('/unit_','unit ')
      disp_name = disp_name.replace('bottleneck_v1','add')
      
#       idx = disp_name.find('/')
#       if idx != -1:
#         disp_name = disp_name[idx+1:]
      disp_name = re.sub(r'block[0-9]','', disp_name)
      disp_name = disp_name.replace('/',' / ')

      layer_names.append(disp_name)
      xs.append( i+1 )    
      Kfrac = get_Kfrac(layer, K)    
      ys.append( Kfrac )

      
    fig, ax = plt.subplots(1,1,figsize=(20,5))
    
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
    width = end-start + 1
    rectangles['RPN'] = Rectangle((start-0.5, ymin), width, height, facecolor='yellow', alpha=0.5)
       
    for r in rectangles:
        ax.add_artist(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width()/2.0
        cy = ry + rectangles[r].get_height()/2.0
        ax.annotate(r, (cx, cy), color='k', weight='bold', fontsize=14, ha='center', va='center')
    
    ax.plot(xs, ys,'o-')
    ax.set_ylim(ymin=ymin)
    ax.set_xlim(xmin=0, xmax=len(xs)+1)
        
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:.0%}'.format(x)))

#     plt.xticks(xs, layer_names, rotation=45)
    plt.xticks(xs, layer_names, rotation='vertical')
    plt.ylabel('$K_{frac}$', fontsize=18)
    plt.xlabel('layer', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.3)
    plt.show()  
     
    


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
