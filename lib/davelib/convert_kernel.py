'''
Created on 2 Jul 2017

@author: david
'''
import numpy as np
import os, cv2, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import os.path
import pickle as pi
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from model.config import cfg
from model.test import _get_blobs
from davelib.compression_stats import CompressionStats
from davelib.profile_stats import ProfileStats
from davelib.base_net import BaseNetWrapper
from davelib.separable_net import SeparableNet
from davelib.pluripotent_net import PluripotentNet
from davelib.layer_name import * 
from davelib.optimise_compression import OptimisationResults, plot_results_from_file
from davelib.utils import show_all_variables, show_all_nodes, colour, timer
from davelib.optimisation_manager import OptimisationManager
from davelib.compressed_net_description import * 


from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.ops import math_ops, array_ops, control_flow_ops
from tensorflow.python.framework.test_util import TensorFlowTestCase

sys.path.append('/home/david/Project/35_tf-faster-rcnn/tools')


def get_blobs(im_names):
    blobs_list = []
    for im_name in im_names:
      im_file = os.path.join(cfg.DATA_DIR, 'demo', im_name)
      im = cv2.imread(im_file)
      blobs, im_scales = _get_blobs(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs['data']
      blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
      blobs_list.append(blobs)
    return blobs_list
        
    
class ExperimentController(TensorFlowTestCase):

  def __init__(self, base_net, sess, saved_model_path, tfconfig, stats_file_suffix):
#     show_all_variables(True, base_net.get_scope())
    self._compressed_layers = None
    self._base_net = base_net
    self._base_net_wrapper = BaseNetWrapper(base_net)
    self._sess = sess
    self._saved_model_path = saved_model_path
    self._tfconfig = tfconfig
#     im_names = ['000456.jpg', '000542.jpg']
#     im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    im_names = ['000456.jpg']
    
    self._blobs_list = get_blobs(im_names)
#     self._final_layers = [LayerName('cls_score')]
    self._final_layers = [LayerName('cls_prob'), LayerName('bbox_pred'), 
                          LayerName('cls_score'), LayerName('rois')]
#     self._final_layers = [LayerName('cls_prob'), LayerName('bbox_pred'), LayerName('rois')]
    
    if os.path.isfile('base_outputs.pi') and os.path.isfile('base_profile_stats.pi'):
#     if False and os.path.isfile('base_outputs.pi') and os.path.isfile('base_profile_stats.pi'):
      self._base_outputs_list = pi.load( open( 'base_outputs.pi', "rb" ) )
      self._base_profile_stats = pi.load( open( 'base_profile_stats.pi', "rb" ) )
    else:
      self._base_outputs_list, run_metadata_list = base_net.get_outputs_multi_image(self._blobs_list,
                                         self._final_layers, sess)
      self._base_profile_stats = ProfileStats(run_metadata_list, tf.get_default_graph())
      pi.dump( self._base_outputs_list, open( 'base_outputs.pi', "wb" ) )
      pi.dump( self._base_profile_stats, open( 'base_profile_stats.pi', "wb" ) )
    
    self._base_variables = tf.global_variables()
    self._all_Kmaxs_dict = calc_all_Kmaxs()
    self._compression_stats = CompressionStats(filename_suffix=stats_file_suffix, 
                                               load_from_file=False)
    self.get_var_dicts()
    
  def get_var_dicts(self):  
    self._all_comp_weights_dict = {}
    self._comp_bn_vars_dict = {}
    self._comp_bias_vars_dict = {}
    names_to_vars = {}
    with self._sess.as_default():
      with tf.variable_scope(self._base_net._resnet_scope, reuse=True):
        for layer_name in get_all_compressible_layers():
          weights = tf.get_variable(layer_name.layer_weights())
          names_to_vars[layer_name] = weights
          
          for bn_type in ['beta','gamma','moving_mean','moving_variance']:
            bn_var_name = layer_name+'/BatchNorm/'+bn_type
            try:
              bn_var = tf.get_variable(bn_var_name)
            except ValueError:
              continue #means this compressed layer doesn't have BatchNorm
            names_to_vars[bn_var_name] = bn_var
            
          bias_var_name = layer_name+'/biases'
          try:
            bias_var = tf.get_variable(bias_var_name)
          except ValueError:
            continue #means this compressed layer doesn't have biases
          names_to_vars[bias_var_name] = bias_var
          
      names = list(names_to_vars.keys())
      vars_ = list(names_to_vars.values())
      tensor_values = self._sess.run( vars_ )  
      
      for name, tensor_value in zip(names, tensor_values):
        if 'BatchNorm' in name:
          self._comp_bn_vars_dict[name] = tensor_value
        elif 'biases' in name:
          self._comp_bias_vars_dict[name] = tensor_value
        else:
          self._all_comp_weights_dict[name] = tensor_value
  
  def _get_next_model(self, layer_K_dict, efficiency_dict, performance_dict, 
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
        K_old = layer_K_dict[layer]
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
            CompressionStep(layer_with_min_grad, K_old, K_new)
            
  
  def _get_next_Kfrac(self, Kfrac):
    print('All layers Kfrac=%.3f'%(Kfrac))
    return self.build_net_desc([Kfrac]), None
    
  def run_optimise_mgr(self, max_iter, stats_file_suffix):
    opt_mgr = OptimisationManager(stats_file_suffix, self)  
    opt_mgr.run_search(max_iter)
    
  def optimise_with_plurinet(self, max_iter, stats_file_suffix):
    guide_compression_stats = CompressionStats(filename_suffix=stats_file_suffix, 
                                load_from_file=True, all_Kmaxs_dict=self._all_Kmaxs_dict)

    guide_compression_stats.add_missing_mAP_deltas( self.get_mAP_base_net([100]) )
 
    stats_dict = guide_compression_stats.build_label_layer_K_dict()
#     efficiency_label = 'float_ops_frac_delta'
    efficiency_label = 'output_bytes_frac_delta'
#     efficiency_label = 'param_output_bytes_frac_delta'
    performance_label = 'mAP_10_top150_delta'
    guide_performance_label = 'mAP_100_top150_delta'
#     performance_label = 'diff_mean_cls_score'
    perf_metric_increases_with_degradation = False
#     performance_label = 'diff_mean_cls_prob'
    use_simple = False #if false use the compound greedy serach

    compressed_layers = get_all_compressible_layers()
    compressed_layers = remove_all_conv_1x1(compressed_layers)
    comp_layer_label='noconv1x1'
#     compressed_layers = keep_only_conv_not_1x1(compressed_layers)
#     compressed_layers = [LayerName('conv1')]
#     comp_layer_label='conv1'
#     compressed_layers = [LayerName('block4/unit_3/bottleneck_v1/conv2')]
#     comp_layer_label='b4_u3_conv2'
    Kfracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    Kfracs_label='0.1-1.0'
#     Kfracs = [0.5]
    num_imgs_list=[10]
    output_layers = []
#     output_layers = self._final_layers
    old_performance_metric = 0

    filename = 'pluri_net_' + Kfracs_label + '_' + comp_layer_label

    pluri_net, mAP_base_net = self.init_plurinet(num_imgs_list, compressed_layers, Kfracs, filename)
    
    efficiency_dict = stats_dict[efficiency_label] #layer->K
    performance_dict = stats_dict[guide_performance_label]
    compression_stats = CompressionStats(load_from_file=False)
 
    efficiency_dict_keys = remove_all_conv_1x1(efficiency_dict.keys())
    efficiency_dict_keys.remove('bbox_pred') #must remove or it will always 'win' as has no effect on cls_score
    efficiency_dict = { k: efficiency_dict[k] for k in efficiency_dict_keys }
#     efficiency_dict = remove_all_except_conv2_first_conv1(efficiency_dict)

    def remove_positive_delta_compressions(efficiency_dict):
      new_dict = {}
      for k, d in efficiency_dict.items(): 
        d2 = {K: val for K, val in d.items() if val<0}
        if len(d2) > 0:
          new_dict[k] = d2
      return new_dict

    efficiency_dict = remove_positive_delta_compressions(efficiency_dict)
    cum_efficiency_delta = 0 
    opt_results = [] 
    layer_K_dict = {} 
    
    for itern in range(max_iter):#each iteration compresses 1 layer (a bit more)
      print(str(itern+1), end=' ')
      layer_K_dict, expected_perf_delta, expected_efficiency_delta, net_change = \
                    self._get_next_model(layer_K_dict, efficiency_dict, performance_dict, 
                                         perf_metric_increases_with_degradation, use_simple)
#     for Kfrac in reversed(Kfracs):
#       layer_K_dict, expected_perf_delta = self._get_next_Kfrac(Kfrac)
#       
      net_desc = CompressedNetDescription(layer_K_dict)
      pluri_net.set_active_path_through_net(net_desc, self._sess)
      
      pluri_net.run_performance_analysis(net_desc, self._blobs_list, self._sess, self._base_outputs_list, 
                                     output_layers, compression_stats, 
                                     self._base_profile_stats, mAP_base_net, num_imgs_list, 
                                     run_profile_stats=False)

      new_performance_metric = compression_stats.get(net_desc, performance_label)
      actual_perf_delta = new_performance_metric - old_performance_metric
      print('\t%s=%f'%(performance_label,new_performance_metric))
      
      opt_results.append( OptimisationResults(expected_efficiency_delta, actual_perf_delta, expected_perf_delta,
                          net_desc, net_change, performance_label, efficiency_label) )

      cum_efficiency_delta += expected_efficiency_delta
      print('\t%s=%f'%(efficiency_label,cum_efficiency_delta))
      if expected_perf_delta:
        print('\texpected/actual \u0394perf=%.3f / %.3f'%(expected_perf_delta,actual_perf_delta))
      old_performance_metric = new_performance_metric
      
      pi.dump( opt_results, open( 'opt_results_output_bytes_clsscore', "wb" ) )

    print('Final optimised net description:\n' + str(net_desc))
    
    type_labels = ['float_ops_frac_delta','float_ops_count_delta',
                   'output_bytes_frac_delta','output_bytes_count_delta',
                   'parameters_frac_delta','parameters_count_delta']
    compression_stats.plot_data_type_by_Kfracs(type_labels)
  
  def run_split_net_exp(self, num_imgs, ):
    compressed_layers_all = get_all_compressible_layers()
    compressed_layers_all = keep_only_conv_not_1x1(compressed_layers_all)
    layer_idxs = [0]
    mAP_base_net = self._base_net_wrapper.mAP(num_imgs, cfg.TEST.RPN_POST_NMS_TOP_N, self._sess)
    
    def run_this_split(compressed_layers):    
      scope_idx=1
      for l, layer_name in enumerate(compressed_layers):
        if l not in layer_idxs:
          continue
        self._compressed_layers = compressed_layers
        
        Kfracs = np.arange(0.7, 1.0, 0.05)
#         Kfracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#         Kfracs = [0.5]
        for Kfrac in Kfracs:
          self._sess.close() #restart session to free memory and to reset profile stats
          tf.reset_default_graph()
          self._sess = tf.Session(config=self._tfconfig) 
  
          net_desc = self.build_net_desc(Kfrac)  
          
          sep_net = SeparableNet(scope_idx, self._base_net, self._sess, self._saved_model_path, 
                                 self._all_comp_weights_dict, net_desc, self._base_variables)
          
          sep_net.run_performance_analysis(self._blobs_list, self._sess, self._base_outputs_list, 
                                           self._final_layers, self._compression_stats, 
                                           self._base_profile_stats, mAP_base_net, num_imgs,
                                           run_profile_stats=False)
          self._compression_stats.set(net_desc, 'Kfrac', Kfrac)
          self._compression_stats.save()
          print(layer_name + ' Kfrac=' + str(Kfrac) + ' complete')
          scope_idx += 1
  #         show_all_variables(True, sep_net._net_sep.get_scope())

    
    self._final_layers = [LayerName('rois')]
    compressed_layers = remove_all_layers_after_rois(compressed_layers_all)
    run_this_split(compressed_layers)
    
    self._final_layers = [LayerName('cls_prob'), LayerName('bbox_pred')]
    num_imgs = 0 #stops calc of mAP
    compressed_layers = remove_all_layers_before_rois(compressed_layers_all)
    run_this_split(compressed_layers)
    
  def get_mAP_base_net(self, num_imgs_list):
    if len(num_imgs_list) > 0:
      mAP_dict = self._base_net_wrapper.mAP(num_imgs_list, cfg.TEST.RPN_POST_NMS_TOP_N, self._sess)
      max_num_imgs = sorted(num_imgs_list)[-1]
      mAP_base_net = mAP_dict[max_num_imgs]
      
      base_net_desc = CompressedNetDescription({})
      for num_imgs, mAP in mAP_dict.items():
        self._compression_stats.set(base_net_desc, 'mAP_%d_top%d'%(num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N), mAP)
      self._compression_stats.save()
    else:
      mAP_base_net = 0
    return mAP_base_net

  def init_plurinet(self, num_imgs_list, compressed_layers, Kfracs, filename):
    mAP_base_net = self.get_mAP_base_net(num_imgs_list)    
    self._compressed_layers = compressed_layers
    net_desc_pluri = build_pluri_net_desc(Kfracs, compressed_layers)  
      
    self._sess.close() #restart session to free memory and to reset profile stats
    tf.reset_default_graph()
    self._sess = tf.Session(config=self._tfconfig) 
     
    pluri_net = PluripotentNet(self._base_net, self._sess, self._saved_model_path, 
                             self._all_comp_weights_dict, self._comp_bn_vars_dict, self._comp_bias_vars_dict, 
                             net_desc_pluri, self._base_variables, filename=filename)
      
    return pluri_net, mAP_base_net
      
  def mAP_for_net(self, net_desc, num_imgs):
    self._sess.close() #restart session to free memory and to reset profile stats
    tf.reset_default_graph()
    self._sess = tf.Session(config=self._tfconfig) 
    
    num_imgs_list = [num_imgs]
    sep_net = SeparableNet(self._base_net, self._sess, self._saved_model_path, 
                         self._all_comp_weights_dict, self._comp_bn_vars_dict, 
                         self._comp_bias_vars_dict, net_desc, self._base_variables)
    
    mAP_base_net = self.get_mAP_base_net(num_imgs_list)  
    compression_stats = CompressionStats(load_from_file=False)
    
    with timer('run_performance_analysis'):
      sep_net.run_performance_analysis(net_desc, self._blobs_list, self._sess, self._base_outputs_list, 
                                     [], compression_stats, 
                                     self._base_profile_stats, mAP_base_net, 
                                     num_imgs_list, run_profile_stats=False)

    mAP = compression_stats.get(net_desc, 'mAP_%d_top%d'%(num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N))
    mAP_delta = compression_stats.get(net_desc, 'mAP_%d_top%d_delta'%
                                                (num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N))
    return mAP

    
  def run_exp(self, num_imgs_list):
    compressed_layers = get_all_compressible_layers()
    compressed_layers = remove_all_conv_1x1(compressed_layers)
    compressed_layers = remove_layers_after_block3(compressed_layers)
#     compressed_layers = keep_only_conv_not_1x1(compressed_layers)
#     compressed_layers = remove_bottleneck_shortcut_layers(compressed_layers)
#     compressed_layers = remove_bottleneck_not_unit1(compressed_layers)
#     compressed_layers = [LayerName('block4/unit_3/bottleneck_v1/conv2/weights')]
#     compressed_layers = [LayerName('rpn_conv/3x3')]
#     compressed_layers = [LayerName('conv1/weights')]
#     compressed_layers = [LayerName('block4/unit_3/bottleneck_v1/conv2'), LayerName('rpn_conv/3x3')]
#     print(compressed_layers)
    self._final_layers = []
    
    #     Ks = range(1,11)
#     layer_idxs = range(7)
    layer_idxs = [0]

    mAP_base_net = self.get_mAP_base_net(num_imgs_list)    
              
    scope_idx=1
    for l, layer_name in enumerate(sorted(compressed_layers)):
#       if l not in layer_idxs:
#         continue
#       self._compressed_layers = [layer_name]
      self._compressed_layers = compressed_layers
      
      Kfracs = [0.5]
#       Kfracs = np.arange(0.01, 1.0, 0.025)
#       Kfracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#       Ks = self.get_Ks(layer_name, Kfracs)

      use_plurinet = False
      if use_plurinet:
        self._sess.close() #restart session to free memory and to reset profile stats
        tf.reset_default_graph()
        self._sess = tf.Session(config=self._tfconfig) 
       
        net_desc_pluri = build_pluri_net_desc(Kfracs, compressed_layers)  
        
        comp_layer_label='block1-3'
        Kfracs_label='0.5'
#         Kfracs_label='0.1-1.0'
        filename = 'pluri_net_' + Kfracs_label + '_' + comp_layer_label

        pluri_net = PluripotentNet(self._base_net, self._sess, self._saved_model_path, 
                                 self._all_comp_weights_dict, self._comp_bn_vars_dict, 
                                 self._comp_bias_vars_dict, 
                                 net_desc_pluri, self._base_variables, filename)
        
      for Kfrac in Kfracs:
        net_desc = build_net_desc([Kfrac], compressed_layers)  
        
        if use_plurinet:
          pluri_net.run_performance_analysis(net_desc, self._blobs_list, self._sess, self._base_outputs_list, 
                                             self._final_layers, self._compression_stats, 
                                             self._base_profile_stats, mAP_base_net, num_imgs_list)
        else:
#           show_all_nodes(True)
          self._sess.close() #restart session to free memory and to reset profile stats
          tf.reset_default_graph()
          self._sess = tf.Session(config=self._tfconfig) 
            
          sep_net = SeparableNet(self._base_net, self._sess, self._saved_model_path, 
                                 self._all_comp_weights_dict, self._comp_bn_vars_dict, 
                                 self._comp_bias_vars_dict, net_desc, self._base_variables)
          
          with timer('run_performance_analysis'):
            sep_net.run_performance_analysis(net_desc, self._blobs_list, self._sess, self._base_outputs_list, 
                                           self._final_layers, self._compression_stats, 
                                           self._base_profile_stats, mAP_base_net, 
                                           num_imgs_list)
        self._compression_stats.save('no1x1conv_0.1-1.0')
        print(layer_name + ' Kfrac=' + str(Kfrac) + ' complete')
        scope_idx += 1

   
def calc_reconstruction_errors(base_net, sess, saved_model_path, tfconfig):
  
  exp_controller = ExperimentController(base_net, sess, saved_model_path, tfconfig, '16')
  exp_controller.run_optimise_mgr(max_iter=5000, stats_file_suffix='allLayersKfrac0.1_1.0')
#   exp_controller.run_exp(num_imgs_list=[])
#   exp_controller.run_exp(num_imgs_list=[25])
#   exp_controller.run_exp(num_imgs_list=[5,10,25,50,100,250,500,1000,2000,4952])
#   exp_controller.run_split_net_exp(num_imgs=100)
#   exp_controller.optimise_with_plurinet(max_iter=50,stats_file_suffix='allLayersKfrac1.0')
#   exp_controller.optimise_with_plurinet(max_iter=10,stats_file_suffix='allLayersKfrac0.1_1.0')
#   exp_controller.optimise_with_plurinet(max_iter=50,stats_file_suffix='no1x1conv_0.1-1.0')
  