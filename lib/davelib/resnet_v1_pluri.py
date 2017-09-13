'''
Created on 2 Jul 2017

@author: david
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import array_ops
# from tensorflow.contrib.slim import arg_scope
from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import utils


from nets.resnet_v1 import resnetv1
from nets.resnet_v1 import resnet_arg_scope
from davelib.layer_name import LayerName
from davelib.resnet_v1_sep import *
from davelib.utils import *


class resnetv1_pluri(resnetv1_sep):
    
  def __init__(self, batch_size, num_layers, base_weights_dict, net_desc, sess=None):
    resnetv1_sep.__init__(self, batch_size, num_layers, base_weights_dict, net_desc)
    self._sess = sess
    self._K_by_layer_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                           value_dtype=tf.int64, default_value=-1)

#   @add_arg_scope
  def bottleneck(self,
                 inputs,
                 depth,
                 depth_bottleneck,
                 stride,
                 rate=1,
                 outputs_collections=None,
                 scope=None):
    """Bottleneck residual unit variant with BN after convolutions.
   
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
   
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
   
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
   
    Returns:
      The ResNet unit's output.
    """
    with variable_scope.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
      depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
      if depth == depth_in:
        shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
      else:
        layer_name = LayerName(sc.name + '/shortcut', 'net_layer')
        if layer_name in self._net_desc:
          shortcut = self.separate_1x1_conv_layer(inputs, depth, stride, layer_name, scope='shortcut')
        else:
          shortcut = layers.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None,
                                   scope='shortcut')
        
      layer_name = LayerName(sc.name + '/conv1', 'net_layer')
      if layer_name in self._net_desc:
        residual = self.separate_1x1_conv_layer(inputs, depth_bottleneck, 1, layer_name, scope='conv1')
      else:
        residual = layers.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
       
      layer_name = LayerName(sc.name + '/conv2', 'net_layer')
      if layer_name in self._net_desc:
        residual = self.separate_conv_layer(residual, depth_bottleneck, 3, stride, 
                                            rate=rate, layer_name='conv2', full_layer_name=layer_name)
      else:
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, 
                                            rate=rate, scope='conv2')

      layer_name = LayerName(sc.name + '/conv3', 'net_layer')
      if layer_name in self._net_desc:
        residual = self.separate_1x1_conv_layer(residual, depth, 1, layer_name, scope='conv3')
      else:
        residual = layers.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
   
      output = nn_ops.relu(shortcut + residual)
   
      return utils.collect_named_outputs(outputs_collections, sc.name, output)

  def separate_1x1_conv_layer(self, inputs, num_outputs, stride, layer_name, scope):
    int_scope = scope + '_sep'
    with arg_scope(
      [layers.conv2d],
      trainable=False,
      normalizer_fn=None,
      normalizer_params=None,
      biases_initializer=None,
      biases_regularizer=None): #make first layer clean, no BN no biases no activation func

      K = self._net_desc[layer_name]
      intermediate = layers.conv2d(inputs, K[0], [1, 1], stride=1, scope=int_scope)
     
    with arg_scope(
      [layers.conv2d],
      trainable=False): #make second layer with BN but with no biases
      net = layers.conv2d(intermediate, num_outputs, [1, 1], stride=stride, scope=scope)
    return net

  def rpn_convolution(self, net_conv4, is_training, initializer):
    layer_name = 'rpn_conv/3x3'
    uncompressed_net = None
    
    def uncompressed_func():
      nonlocal uncompressed_net
      if uncompressed_net is None:#need this as irritatingly the tf.case() calls it twice
        uncompressed_net = super(resnetv1_sep, self).rpn_convolution(net_conv4, is_training, initializer)
      return uncompressed_net

    if layer_name not in self._net_desc:
      net = uncompressed_func()
      return net

    def build_layer(K):
      with arg_scope(
        [slim.conv2d],
        trainable=False,
        normalizer_fn=None,
        normalizer_params=None,
        biases_initializer=None,
        biases_regularizer=None): #make first layer clean, no BN no biases no activation func
       
        layer1_name = LayerName(layer_name + '_sep_K'+str(K))
        net = slim.conv2d(net_conv4, K, [3, 1], trainable=is_training, weights_initializer=initializer,
                          scope=layer1_name)
  
        layer2_name = LayerName(layer_name + '_K'+str(K))
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with arg_scope(
          [slim.conv2d],
          trainable=False,
          normalizer_fn=None,
          normalizer_params=None): #make second layer no BN but with biases
          net = slim.conv2d(net, 512, [1, 3], trainable=is_training, weights_initializer=initializer,
                            scope=layer2_name)
      return net
    
    K_active = self._K_by_layer_table.lookup( tf.constant(layer_name, dtype=tf.string) )
    case_dict = {}    
    Ks = self._net_desc[layer_name]
    for K in Ks:
      match_cond = tf.equal(K_active, tf.constant(K, dtype=tf.int64))
      case_dict[match_cond] = lambda K_=K: build_layer(K_)
    
    return tf.case(case_dict, default=uncompressed_func, exclusive=True)
  
  def fully_connected(self, input_, num_outputs, is_training, initializer, layer_name):
    uncompressed_net = None
    
    def uncompressed_func():
      nonlocal uncompressed_net
      if uncompressed_net is None:#need this as irritatingly the tf.case() calls it twice
        uncompressed_net = super(resnetv1_sep, self).fully_connected(input_, num_outputs, 
                                                      is_training, initializer, layer_name)
      return uncompressed_net

    if layer_name not in self._net_desc:
      net = uncompressed_func()
      return net

    def build_layer(K):
      with arg_scope(
        [slim.fully_connected],
        trainable=False,
        normalizer_fn=None,
        normalizer_params=None,
        biases_initializer=None,
        biases_regularizer=None): #make first layer clean, no BN no biases no activation func
       
        layer1_name = LayerName(layer_name + '_sep_K'+str(K))
        net = slim.fully_connected(input_, K, weights_initializer=initializer,
                                  trainable=is_training, activation_fn=None, scope=layer1_name)
  
        layer2_name = LayerName(layer_name + '_K'+str(K))
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with arg_scope(
          [slim.fully_connected],
          trainable=False,
          normalizer_fn=None,
          normalizer_params=None): #make second layer no BN but with biases
          net = slim.fully_connected(net, num_outputs, weights_initializer=initializer,
                                  trainable=is_training, scope=layer2_name)
        return net
    
    K_active = self._K_by_layer_table.lookup( tf.constant(layer_name, dtype=tf.string) )
    case_dict = {}    
    Ks = self._net_desc[layer_name]
    for K in Ks:
      match_cond = tf.equal(K_active, tf.constant(K, dtype=tf.int64))
      case_dict[match_cond] = lambda K_=K: build_layer(K_)
    
    return tf.case(case_dict, default=uncompressed_func, exclusive=True)

       
  def separate_conv_layer(self, inputs, num_output_channels, kernel_size, stride, rate,
                          layer_name, full_layer_name):
    def build_layer(K):
      with arg_scope(
        [slim.conv2d],
        weights_regularizer=None,
        weights_initializer=None,
        trainable=False,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None,
        biases_initializer=None): #make first layer clean, no BN no biases no activation func
  
        layer1_name = LayerName(layer_name + '_sep_K'+str(K))
        net = conv2d_same(inputs, K, kernel_size=(kernel_size,1), stride=[stride,1],
                           scope=layer1_name)
      
        layer2_name = LayerName(layer_name + '_K'+str(K))
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = conv2d_same(net, num_output_channels, kernel_size=(1,kernel_size), 
                          stride=[1,stride], scope=layer2_name)
      return net

    uncompressed_net = None
    def uncompressed_func():
      nonlocal uncompressed_net
      if uncompressed_net is None:#need this as irritatingly the tf.case() calls it twice
        uncompressed_net = resnet_utils.conv2d_same(inputs, num_output_channels, kernel_size, 
                                                stride=stride, scope=layer_name)
      return uncompressed_net

    K_active = self._K_by_layer_table.lookup( tf.constant(full_layer_name, dtype=tf.string) )
    case_dict = {}    
    Ks = self._net_desc[full_layer_name]
    for K in Ks:
      match_cond = tf.equal(K_active, tf.constant(K, dtype=tf.int64))
      case_dict[match_cond] = lambda K_=K: build_layer(K_)
    
    return tf.case(case_dict, default=uncompressed_func, exclusive=True)


  def _check_proposed_active_path(self, net_desc):
    #the active path must be a subset of the possible layers and Ks
    for layer, Ks in net_desc.items():
      active_K = Ks[0]
      active_K_possible = False
      if layer in self._net_desc:
        Ks_ = self._net_desc[layer]
        for K in Ks_:
          if active_K == K:
            active_K_possible = True
            break
      if not active_K_possible:
        raise ValueError('%s with active_K=%d isn\'t possible'%(layer, active_K))
    
  def set_active_path_through_net(self, net_desc, sess, print_path=False):
    self._check_proposed_active_path(net_desc)
       
    self._active_Ks = np.zeros( len(net_desc) ) 
    keys_tensor, _ = self._K_by_layer_table.export()
    layers = sess.run([keys_tensor])
  
    #first set to -1 all the entries in the table that aren't in this net_desc
    keys = []
    values=[]
    for tbl_key in layers[0]:
      if tbl_key not in net_desc:
        keys.append(tbl_key)
        values.append(-1)
          
    #now set all entries in net_desc in the table
    for layer, K in net_desc.items():
      keys.append(layer)
      values.append(K[0])
    self._K_by_layer_table.insert(tf.constant(keys, dtype=tf.string), 
                                  tf.constant(values, dtype=tf.int64)).run(session=sess)
    
    if print_path:
      print('Active path through net:')
  #     s = self._K_by_layer_table.size().eval(session=sess)
      keys, values = self._K_by_layer_table.export()
      layers, Ks = sess.run([keys, values])
      for layer, K in sorted(zip(layers, Ks)):
        print('\t%s\tK=%d'%(layer.decode("utf-8"), K))

    
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}
    
    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois
    
      
