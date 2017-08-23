# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os.path as osp

from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.python.client import timeline

from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
from tensorflow.contrib.layers.python.layers import utils

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.RESNET.BN_TRAIN,
    'updates_collections': ops.GraphKeys.UPDATE_OPS,
    'fused': True
  }

  with arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
#       normalizer_fn=None,
#       normalizer_params=None):
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc
    


class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers
    self.bottleneck_func = resnet_v1.bottleneck
    self._end_points_collection = self._resnet_scope + '_end_points'
      
    
  def get_scope(self):
    return self._resnet_scope

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
#       with arg_scope(
#         [slim.conv2d],
#         weights_regularizer=None,
#         weights_initializer=None,
#         trainable=False,
# #         activation_fn=nn_ops.relu,
#         activation_fn=None,
#         normalizer_fn=None,
#         normalizer_params=None,
#         biases_initializer=None): #make first layer clean, no BN no biases

      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      self._predictions[self._resnet_scope+'/conv1'] = net

#       end_points_collection = self._resnet_scope + '_end_points'
      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/conv1', 
                                  net)
      
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

      return net

  def resnet_v1_block(self, scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.
  
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
  
    Returns:
      A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, self.bottleneck_func, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
#     bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [
          self.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          self.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          self.resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
          self.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)
      ]
    elif self._num_layers == 101:
      blocks = [
          self.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          self.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          self.resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
          self.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)
      ]
#       blocks = [
#         resnet_utils.Block('block1', bottleneck,
#                            [(256, 64, 1)] * 2 + [(256, 64, 2)]),
#         resnet_utils.Block('block2', bottleneck,
#                            [(512, 128, 1)] * 3 + [(512, 128, 2)]),
#         # Use stride-1 for the last conv4 layer
#         resnet_utils.Block('block3', bottleneck,
#                            [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
#         resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
#       ]
    elif self._num_layers == 152:
      blocks = [
          self.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          self.resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
          self.resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
          self.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError
    
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS == 3:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    elif cfg.RESNET.FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = self.rpn_convolution(net_conv4, is_training, initializer)
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
      else:
        raise NotImplementedError

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      
      cls_score = self.fully_connected(fc7, self._num_classes, is_training, initializer, 'cls_score')
#       cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
#                                        trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = self.fully_connected(fc7, self._num_classes * 4, is_training, initializer, 'bbox_pred')
#       bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
#                                        trainable=is_training,
#                                        activation_fn=None, scope='bbox_pred')

      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/rois', rois)
      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/fc7', fc7)
      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/cls_score', cls_score)
      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/cls_prob', cls_prob)
      utils.collect_named_outputs(self._end_points_collection, self._resnet_scope+'/bbox_pred', bbox_pred)
    
    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

#   def separate_1x1_conv_layer(self, inputs, num_outputs, layer_name):
#     return layers.conv2d(inputs, num_outputs, [1, 1], stride=1, scope=layer_name)

  def rpn_convolution(self, net_conv4, is_training, initializer):
    return slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
     
  def fully_connected(self, input_, num_outputs, is_training, initializer, layer_name):
    return slim.fully_connected(input_, num_outputs, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope=layer_name)

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'], 
                           tf.reverse(conv1_rgb, [2])))
        
  def get_outputs_multi_image(self, blobs_list, output_layers, sess):
    outputs_list = []
    run_metadata_list = []
    
    for blobs in blobs_list:
      outputs, run_metadata = self.get_outputs(blobs, output_layers, sess)
      outputs_list.append(outputs)
      run_metadata_list.append(run_metadata)
    return outputs_list, run_metadata_list

  def get_outputs(self, blobs, output_layers, sess):
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
    
    # Create the Timeline object, and write it to a json
#     tl = timeline.Timeline(run_metadata.step_stats)
#     ctf = tl.generate_chrome_trace_format()
#     with open('timeline.json', 'w') as f:
#       f.write(ctf)

#     outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'graph_defs'))      
#     writer = tf.summary.FileWriter(logdir=outdir, graph=sess.graph)
#     writer.add_run_metadata(run_metadata, 'step1')
#     writer.flush()
    
    return outputs, run_metadata
  
  
def remove_net_suffix(input_str, net_root):
  # nasty function to convert e.g. resnet_v1_101_2/block2/unit_1/bottleneck_v1/
  # to                             resnet_v1_101/block2/unit_1/bottleneck_v1/
  # hack to deal with the fact tf adds suffix to scope original name
  idx = input_str.find(net_root)
  if idx == -1:
    return input_str
  else:
    idx = input_str.index('/')
    return net_root + input_str[idx:]
  
  
  
  
  
    