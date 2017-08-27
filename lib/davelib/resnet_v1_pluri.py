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
from tensorflow.python.ops import nn_ops
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
    
  def __init__(self, scope_idx, batch_size, num_layers, base_weights_dict, net_desc):
    resnetv1.__init__(self, batch_size, num_layers)
    self._resnet_scope = 'resnet_v1_sep%d_%d' % (scope_idx, num_layers)  
    self._base_weights_dict = base_weights_dict
    self._net_desc = net_desc
    self.bottleneck_func = self.bottleneck
    self._end_points_collection = self._resnet_scope + '_end_points'
    self._K_active_placeholder = tf.placeholder(tf.int32, shape=(len(net_desc)), name='K_active_placeholder')
    self._layer_active_placeholder = tf.placeholder(tf.string, shape=(len(net_desc)), name='layer_active_placeholder')
    self._K_by_layer_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(self._layer_active_placeholder, 
                                                  self._K_active_placeholder), -1, 
                                                         name='K_by_layer_table')

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
      intermediate = layers.conv2d(inputs, K, [1, 1], stride=1, scope=int_scope)
     
    with arg_scope(
      [layers.conv2d],
      trainable=False): #make second layer with BN but with no biases
      net = layers.conv2d(intermediate, num_outputs, [1, 1], stride=stride, scope=scope)
    return net

  def separate_conv_layer(self, inputs, num_output_channels, kernel_size, stride, rate,
                          layer_name, full_layer_name):

    uncompressed_net = resnet_utils.conv2d_same(inputs, num_output_channels, kernel_size, 
                                                stride=stride, scope=layer_name)
    K_active = self._K_by_layer_table.lookup( tf.constant(layer_name, dtype=tf.string) )

    case_dict = {}
    Ks = self._net_desc[full_layer_name]
    for K in Ks:
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
        
      case_dict[ tf.equal(K_active, tf.constant(K, dtype=tf.int32)) ] = lambda: net
    
#     output = tf.case(case_dict, default=lambda: uncompressed_net, exclusive=True)
    output = net
#     output = uncompressed_net
    return output

  def get_outputs(self, blobs, output_layers, sess, net_desc):
    output_layers.append(LayerName('conv1'))
     
    layers = list(net_desc.keys())
    Ks = list(net_desc.values())
    feed_dict = {self._layer_active_placeholder: layers,
                 self._K_active_placeholder: Ks}
    self._K_by_layer_table.init.run(feed_dict=feed_dict, session=sess)
    K_active = self._K_by_layer_table.lookup( tf.constant(LayerName('conv1'), dtype=tf.string) )
    print('conv1 K_active=%d'%K_active.eval(session=sess))
    
    feed_dict = {self._image: blobs['data'],
                 self._im_info: blobs['im_info'],
                 self._gt_boxes: np.zeros((10,5))}
    fetches = {}
    
    for collection_name in ops.get_all_collection_keys():
      if self._resnet_scope in collection_name:
        collection_dict = utils.convert_collection_to_dict(collection_name)
        for alias, tensor in collection_dict.items():
          alias = remove_net_suffix(alias, self._resnet_scope)
          for output_layer in output_layers:
            if output_layer.net_layer(self._resnet_scope) in alias:
              fetches[output_layer] = tensor
              
    # Run the graph with full trace option
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    outputs = sess.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    return outputs, run_metadata  
      
  def rpn_convolution(self, net_conv4, is_training, initializer):
    layer_name = 'rpn_conv/3x3'

    if layer_name not in self._net_desc:
      return super(resnetv1_sep, self).rpn_convolution(net_conv4, is_training, initializer)

    K = self._net_desc[layer_name]
    layer1_name = LayerName(layer_name + '_sep')
    with arg_scope(
      [slim.conv2d],
      trainable=False,
      normalizer_fn=None,
      normalizer_params=None,
      biases_initializer=None,
      biases_regularizer=None): #make first layer clean, no BN no biases no activation func
     
      net = slim.conv2d(net_conv4, K, [3, 1], trainable=is_training, weights_initializer=initializer,
                        scope=layer1_name)

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      with arg_scope(
        [slim.conv2d],
        trainable=False,
        normalizer_fn=None,
        normalizer_params=None): #make second layer no BN but with biases
        net = slim.conv2d(net, 512, [1, 3], trainable=is_training, weights_initializer=initializer,
                          scope=layer_name)
    return net

  def fully_connected(self, input_, num_outputs, is_training, initializer, layer_name):
    if layer_name not in self._net_desc:
      return super(resnetv1_sep, self).fully_connected(input_, num_outputs, is_training, 
                                                       initializer, layer_name)
    K = self._net_desc[layer_name]
    layer1_name = LayerName(layer_name +'_sep')
    with arg_scope(
      [slim.fully_connected],
      trainable=False,
      normalizer_fn=None,
      normalizer_params=None,
      biases_initializer=None,
      biases_regularizer=None): #make first layer clean, no BN no biases no activation func
     
      net = slim.fully_connected(input_, K, weights_initializer=initializer,
                                trainable=is_training, activation_fn=None, scope=layer1_name)

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      with arg_scope(
        [slim.fully_connected],
        trainable=False,
        normalizer_fn=None,
        normalizer_params=None): #make second layer no BN but with biases
        net = slim.fully_connected(net, num_outputs, weights_initializer=initializer,
                                trainable=is_training, scope=layer_name)
    return net
 
  def build_base(self):
    layer_name = LayerName('conv1')
    if layer_name in self._net_desc:
      with tf.variable_scope(self._resnet_scope, self._resnet_scope):
        N = self._base_weights_dict[layer_name].shape[3]
  
        net = self.separate_conv_layer(self._image, N, 7, 2, rate=None, layer_name=layer_name,
                                       full_layer_name=layer_name)
  
        end_points_collection = self._resnet_scope + '_end_points'
        utils.collect_named_outputs(end_points_collection, self._resnet_scope+'/conv1', net)        
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
    else:
      net = super(resnetv1_sep, self).build_base()
    return net
