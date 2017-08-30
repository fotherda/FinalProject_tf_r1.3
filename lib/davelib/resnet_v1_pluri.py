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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes


from nets.resnet_v1 import resnetv1
from nets.resnet_v1 import resnet_arg_scope
from davelib.layer_name import LayerName
from davelib.resnet_v1_sep import *
from davelib.utils import *


class resnetv1_pluri(resnetv1_sep):
    
  def __init__(self, scope_idx, batch_size, num_layers, base_weights_dict, net_desc, sess=None):
    resnetv1.__init__(self, batch_size, num_layers)
    self._resnet_scope = 'resnet_v1_sep%d_%d' % (scope_idx, num_layers)  
    self._base_weights_dict = base_weights_dict
    self._net_desc = net_desc
    self.bottleneck_func = self.bottleneck
    self._sess = sess
    self._end_points_collection = self._resnet_scope + '_end_points'
#     self._K_active_placeholder = tf.placeholder(tf.int32, shape=(len(net_desc)), name='K_active_placeholder')
#     self._layer_active_placeholder = tf.placeholder(tf.string, shape=(len(net_desc)), name='layer_active_placeholder')
    self._K_by_layer_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                           value_dtype=tf.int64, default_value=-1)
#     self._K_by_layer_table = tf.contrib.lookup.HashTable(
#       tf.contrib.lookup.KeyValueTensorInitializer(self._layer_active_placeholder, 
#                                                   self._K_active_placeholder), -1, 
#                                                           name='K_by_layer_table')
    self._active_Ks_placeholder = tf.placeholder(tf.int32, shape=(len(net_desc)), name='K_active_placeholder')
    self._layer_to_active_Ks_dict = {}
    for i, layer_name in enumerate(net_desc.keys()):
      self._layer_to_active_Ks_dict[layer_name] = i

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

  def build_layer(self, K, net_conv4, is_training, initializer, layer_name):
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

  
  def rpn_convolution(self, net_conv4, is_training, initializer):
    layer_name = 'rpn_conv/3x3'
#     uncompressed_net = super(resnetv1_sep, self).rpn_convolution(net_conv4, is_training, initializer)
#     if layer_name not in self._net_desc: 
#       return uncompressed_net
    uncompressed_net = None
    idx_this_layer = self._layer_to_active_Ks_dict[layer_name]

    case_dict = {}    
    case_list = []    
    nets = []
    Ks = self._net_desc[layer_name]
#     for K in Ks:
# #     for K in reversed(Ks):
#       net = self.build_layer(K, net_conv4, is_training, initializer, layer_name)
#       nets.append(net)  
#           case_list.append( (math_ops.equal(K_active, constant_op.constant(33, dtype=dtypes.int64)),
#                               lambda: net) )
#         case_dict[ math_ops.equal(K_active, constant_op.constant(K, dtype=tf.int64)) ] = lambda: net
#         match_cond = tf.equal(self._active_Ks_placeholder[idx_this_layer], tf.constant(K+3))
#         case_dict[ match_cond ] = lambda: net

#     layer_tensor = tf.constant(layer_name, dtype=tf.string)
#     K_tensor = tf.constant(Ks[0], dtype=tf.int64)
#     self._K_by_layer_table.insert(layer_tensor, K_tensor).run(session=self._sess)
#           
    def f1(): 
      return self.build_layer(Ks[0], net_conv4, is_training, initializer, layer_name)
#       with tf.control_dependencies([nets[0]]):
#         return nets[0]
    def f2(): 
      return self.build_layer(Ks[1], net_conv4, is_training, initializer, layer_name)
    def f3():
      nonlocal uncompressed_net
      if uncompressed_net is None:
        uncompressed_net = super(resnetv1_sep, self).rpn_convolution(net_conv4, is_training, initializer)
      return uncompressed_net
    
    K_active = self._K_by_layer_table.lookup( constant_op.constant(layer_name, dtype=tf.string) )
    match_cond1 = tf.equal(K_active, tf.constant(Ks[0], dtype=tf.int64))
    match_cond2 = tf.equal(K_active, tf.constant(Ks[1], dtype=tf.int64))
#     match_cond1 = tf.equal(self._active_Ks_placeholder[idx_this_layer], tf.constant(Ks[0]))
#     match_cond2 = tf.equal(self._active_Ks_placeholder[idx_this_layer], tf.constant(Ks[1]))
#     cond_result = tf.cond(match_cond, lambda: f1(), lambda: f2())
    case_result = tf.case({match_cond1: lambda: f1(), match_cond2: lambda: f2()}, 
                          default=lambda: f3(), exclusive=True)
    
#     def f1(): return nets[0]
#     def f2(): return nets[1]
#     case_dict[ tf.equal(tf.constant(3),tf.constant(2)) ] = f1
#     case_dict[ tf.equal(tf.constant(2),tf.constant(3)) ] = f2
#     case_list.append( (math_ops.equal(tf.constant(2),tf.constant(3)), f1))
#     case_list.append( (math_ops.equal(tf.constant(2),tf.constant(3)), f2))
#                               lambda: net) )
#     case_dict[ tf.equal(K_active, tf.constant(1, dtype=tf.int64)) ] = f1
#     case_dict[ tf.equal(K_active, tf.constant(2, dtype=tf.int64)) ] = f2
#     match_cond1 = tf.equal(tf.constant(2), tf.constant(3))
#     match_cond1 = tf.equal(self._active_Ks_placeholder[idx_this_layer], tf.constant(3))
#     case_dict[ match_cond1] = f1
#     match_cond2 = tf.equal(tf.constant(1), tf.constant(2))
#     case_dict[ match_cond2 ] = f2
#       
    
#     output = tf.case({tf.equal(K_active, tf.constant(Ks[0] + 1, dtype=tf.int64)): f2,
#                       tf.equal(K_active, tf.constant(Ks[0] + 1, dtype=tf.int64)): f1}, 
#                      exclusive=True)
#     output = control_flow_ops.case(case_list, exclusive=True)
    output = case_result
#     output = control_flow_ops.case(case_dict, exclusive=True)
#     output = tf.case(case_dict, default=lambda: uncompressed_net, exclusive=True)
#     output = net
#     output = uncompressed_net
    return output
  
  def separate_conv_layer(self, inputs, num_output_channels, kernel_size, stride, rate,
                          layer_name, full_layer_name):

    uncompressed_net = resnet_utils.conv2d_same(inputs, num_output_channels, kernel_size, 
                                                stride=stride, scope=layer_name)
    if layer_name not in self._net_desc: 
      return uncompressed_net

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
        
      case_dict[ tf.equal(K_active, tf.constant(K, dtype=tf.int64)) ] = lambda: net
    
    output = tf.case(case_dict, exclusive=True)
#     output = tf.case(case_dict, default=lambda: uncompressed_net, exclusive=True)
#     output = net
#     output = uncompressed_net
    return output
  
#   def set_active_path_through_net(self, net_desc):  
#     self._active_Ks = np.zeros( len(net_desc) )
#     for layer, K in net_desc.items():
#       self._active_Ks[ self._layer_to_active_Ks_dict[layer] ] = K[0]

  def set_active_path_through_net(self, net_desc, sess):   
    self._active_Ks = np.zeros( len(net_desc) ) 
    data = self._K_by_layer_table.export()
    d = sess.run(data)
  
    for layer, K in net_desc.items():
      layer_tensor = tf.constant(layer, dtype=tf.string, shape=[1])
      K_tensor = tf.constant(K[0], dtype=tf.int64, shape=[1])
      self._K_by_layer_table.insert(layer_tensor, K_tensor).run(session=sess)
        
    s = self._K_by_layer_table.size().eval(session=sess)
    data = self._K_by_layer_table.export()
    d = sess.run(data)
#     print(d)
    
  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info,
                 self._active_Ks_placeholder: self._active_Ks}
    
    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois
    
  def get_outputs_multi_image(self, blobs_list, output_layers, sess):
    outputs_list = []
    run_metadata_list = []
#     self.set_active_path_through_net(net_desc, sess)
    
    for blobs in blobs_list:
      outputs, run_metadata = self.get_outputs(blobs, output_layers, sess)
      outputs_list.append(outputs)
      run_metadata_list.append(run_metadata)
    return outputs_list, run_metadata_list


  def get_outputs(self, blobs, output_layers, sess):
#     output_layers.append(LayerName('conv1'))
#     layer = LayerName('block4/unit_3/bottleneck_v1/conv2')
    layer = LayerName('rpn_conv/3x3')
     
    K_active = self._K_by_layer_table.lookup( tf.constant(layer, dtype=tf.string) )
    print('%s K_active=%d'%(layer, K_active.eval(session=sess)))
    
    feed_dict = {self._image: blobs['data'],
                 self._im_info: blobs['im_info'],
                 self._gt_boxes: np.zeros((10,5)),
                 self._active_Ks_placeholder: self._active_Ks}
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
     
      net = slim.fully_connected(input_, K[0], weights_initializer=initializer,
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
