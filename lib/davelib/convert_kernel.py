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
from davelib.layer_name import LayerName
from davelib.compression_stats import CompressionStats, CompressedNetDescription
from collections import OrderedDict
from davelib.profile_stats import ProfileStats
from davelib.base_net import BaseNetWrapper
from davelib.separable_net import SeparableNet
from davelib.pluripotent_net import PluripotentNet
from davelib.layer_name import * 
from davelib.utils import show_all_variables



sys.path.append('/home/david/Project/35_tf-faster-rcnn/tools')
figure_path = '/home/david/host/figures'


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
        

    
class ExperimentController(object):

  def __init__(self, base_net, sess, saved_model_path, tfconfig, stats_file_suffix):
#     show_all_variables(True, base_net.get_scope())
    self._compressed_layers = None
    self._base_net = base_net
    self._base_net_wrapper = BaseNetWrapper(base_net)
    self._sess = sess
    self._saved_model_path = saved_model_path
    self._tfconfig = tfconfig
#     im_names = ['000456.jpg', '000542.jpg']
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
#     im_names = ['000456.jpg']
    
    self._blobs_list = get_blobs(im_names)
#     self._final_layers = [LayerName('cls_score')]
    self._final_layers = [LayerName('cls_prob'), LayerName('bbox_pred'), LayerName('rois')]
#     self._final_layers = [LayerName('cls_score'), LayerName('bbox_pred')]
#     final_layer = LayerName('block3/unit_23/bottleneck_v1/conv3')

    
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

    self._all_Kmaxs_dict = self.calc_all_Kmaxs()
    self._compression_stats = CompressionStats(filename_suffix=stats_file_suffix, 
                                               load_from_file=False, all_Kmaxs_dict=self._all_Kmaxs_dict)
    self._all_comp_weights_dict = {}
    with sess.as_default():
      with tf.variable_scope(base_net._resnet_scope, reuse=True):
        for layer_name in get_all_compressible_layers():
          weights = tf.get_variable(layer_name.layer_weights())
          self._all_comp_weights_dict[layer_name] = weights.eval()
          
           
          bn_weights = tf.get_variable(layer_name.layer_weights())
          self._all_comp_weights_dict[layer_name] = weights.eval()
          
          

  def calc_all_Kmaxs(self):
    d = {}
    for layer in get_all_compressible_layers():
      Kmax = self.calc_Kmax(layer)
      d[layer] = Kmax
    return d
    
  def calc_Kmax(self, layer):
    shape = None
    for v in tf.global_variables():
      if '/'+layer.layer_weights() in v.name:
        shape = v.get_shape().as_list()
        break
    if not shape:
      raise Exception('layer not found') 
        
    if len(shape)==4: #convolutional layer
      H,W,C,N = tuple(shape)
      Kmax = int(C*W*H*N / (C*W + H*N)) # if K > Kmax will have more parameters in sep layer
    elif len(shape)==2: #fully connected layer
      C,N = tuple(shape)
      Kmax = int(C*N / (C + N)) # if K > Kmax will have more parameters in sep layer

#     print('%s %d %d %d'%(layer, C, N, Kmax))

    return Kmax
        
  def get_Ks(self, layer, K_fractions):
#     Kmax = self.get_Kmax(layer)
    Kmax = self._all_Kmaxs_dict[layer]
    
    Ks = []
    for K_frac in K_fractions:
      K = int(K_frac * Kmax)
      if K == 0:
        K = 1
      if K > Kmax:
        K = Kmax
      Ks.append(K)
    return Ks
  
  def build_pluri_net_desc(self, Kfracs):
    K_by_layer_dict = {}
    for layer_name in self._compressed_layers:
      Ks = self.get_Ks(layer_name, Kfracs)
      K_by_layer_dict[layer_name] = Ks
    net_desc = CompressedNetDescription(K_by_layer_dict)
    return net_desc
  
  def build_net_desc(self, Kfrac):
    K_by_layer_dict = {}
    for layer_name in self._compressed_layers:
      K = self.get_Ks(layer_name, [Kfrac])
      K_by_layer_dict[layer_name] = K[0]
    net_desc = CompressedNetDescription(K_by_layer_dict)
    return net_desc
  
  def optimise_for_memory(self, max_iter, stats_file_suffix):
    guide_compression_stats = CompressionStats(filename_suffix=stats_file_suffix, 
                                load_from_file=True, all_Kmaxs_dict=self._all_Kmaxs_dict)
    
    stats_dict = guide_compression_stats.build_label_layer_K_dict()
    objective_label = 'param_bytes_frac_delta'
#     objective_label = 'total_output_bytes_frac_delta'
#     objective_label = 'total_bytes_frac_delta'
#     performance_label = 'diff_mean_cls_score'
    performance_label = 'diff_mean_cls_prob'
    
    objective_dict = stats_dict[objective_label] #layer->K
    performance_dict = stats_dict[performance_label]

    #bbox_pred layer doesn't affect cls_score
    del objective_dict['bbox_pred']
    
    def remove_all_except_conv2_first_conv1(d):  
      filtered = {}
      for k,v in d.items():
        if 'block' not in k and 'conv1' in k:
          filtered[k] = v
        elif 'conv2' in k:
          filtered[k] = v
        elif 'rpn' in k:
          filtered[k] = v
      return filtered

    objective_dict = remove_all_except_conv2_first_conv1(objective_dict)
    
    layer_K_dict = {} 
    
    def get_K_old_new(layer):
      sorted_keys = list(reversed(sorted(objective_dict[layer].keys())))
      if layer in layer_K_dict:
        K_old = layer_K_dict[layer]
        idx = sorted_keys.index( K_old )
        if idx+1 < len(sorted_keys):
          K_new = sorted_keys[idx + 1]
        else:
          K_new = 0
      else:
        K_new = sorted_keys[0] #start with the largest K value
        K_old = 0
      return K_old, K_new

    scope_idx=1
    old_performance_metric = 0
    
    for iter in range(max_iter):#each iteration compresses 1 layer (a bit more)
      grad_dict = {}
      perf_delta_dict = {}
      for layer in objective_dict:
        K_old, K_new = get_K_old_new(layer)  
        if K_new == 0: #max compression reached for this layer
          continue
        elif K_old != 0:
          objective_delta = objective_dict[layer][K_new] - objective_dict[layer][K_old]
          performance_delta = performance_dict[layer][K_new] - performance_dict[layer][K_old]
        else:
          objective_delta = objective_dict[layer][K_new]
          performance_delta = performance_dict[layer][K_new]

        objective_grad = objective_delta / performance_delta
        grad_dict[objective_grad] = layer 
        perf_delta_dict[layer] = performance_delta
        
      grad_min = list(sorted(grad_dict))[0]
      layer_with_min_grad = grad_dict[grad_min]
      K_old, K_new = get_K_old_new(layer_with_min_grad)
      layer_K_dict[layer_with_min_grad] = K_new
      expected_perf_delta = perf_delta_dict[layer_with_min_grad]
      
      self._sess.close() #restart session to free memory and to reset profile stats
      tf.reset_default_graph()
      self._sess = tf.Session(config=self._tfconfig) 

      net_desc = CompressedNetDescription(layer_K_dict)
      
      sep_net = SeparableNet(scope_idx, self._base_net, self._sess, self._saved_model_path, 
                             self._all_comp_weights_dict, net_desc, self._base_variables)
      
      sep_net.run_performance_analysis(self._blobs_list, self._sess, self._base_outputs_list, 
                                       self._final_layers, self._compression_stats, 
                                       self._base_profile_stats, run_profile_stats=False)
      
      new_performance_metric = self._compression_stats.get(net_desc, performance_label)
#       objective_metric = self._compression_stats.get(net_desc, objective_label)
      actual_perf_delta = new_performance_metric - old_performance_metric
      print('%s: K:%d->%d perf=%f'%(layer_with_min_grad,K_old,K_new,new_performance_metric))
      print('expected perf delta=%.3f, actual perf delta=%.3f'%(expected_perf_delta,actual_perf_delta))
      old_performance_metric = new_performance_metric
      
      scope_idx += 1
  
  def run_split_net_exp(self, num_imgs):
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
    
    
  def run_exp(self, num_imgs_list):
    compressed_layers = get_all_compressible_layers()
    compressed_layers = remove_all_except_conv2_first_conv1(compressed_layers)
#     compressed_layers = remove_bottleneck_shortcut_layers(compressed_layers)
#     compressed_layers = remove_bottleneck_not_unit1(compressed_layers)
    
    #     Ks = range(1,11)
#     layer_idxs = range(7)
    layer_idxs = [0]


    mAP_dict = self._base_net_wrapper.mAP(num_imgs_list, cfg.TEST.RPN_POST_NMS_TOP_N, self._sess)
    max_num_imgs = sorted(num_imgs_list)[-1]
    mAP_base_net = mAP_dict[max_num_imgs]
    
    base_net_desc = CompressedNetDescription({})
    for num_imgs, mAP in mAP_dict.items():
        self._compression_stats.set(base_net_desc, 'mAP_%d_top%d'%(num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N), mAP)
    self._compression_stats.save()
              
    scope_idx=1
    for l, layer_name in enumerate(compressed_layers):
      if l not in layer_idxs:
        continue
      self._compressed_layers = [layer_name]
#       self._compressed_layers = compressed_layers
      
#       Kfracs = [1.0]
#       Kfracs = np.arange(0.01, 1.0, 0.025)
#       Kfracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      Kfracs = [0.32,0.34,0.36,0.38]
#       Ks = self.get_Ks(layer_name, Kfracs)

#       self._sess.close() #restart session to free memory and to reset profile stats
#       tf.reset_default_graph()
#       self._sess = tf.Session(config=self._tfconfig) 
#       
#       net_desc_pluri = self.build_pluri_net_desc(Kfracs)  
#       pluri_net = PluripotentNet(scope_idx, self._base_net, self._sess, self._saved_model_path, 
#                                self._all_comp_weights_dict, net_desc_pluri, self._base_variables)
       
      for Kfrac in Kfracs:
        net_desc = self.build_net_desc(Kfrac)  
 
#         pluri_net.run_performance_analysis(net_desc, self._blobs_list, self._sess, self._base_outputs_list, 
#                                            self._final_layers, self._compression_stats, 
#                                            self._base_profile_stats, mAP_base_net, num_imgs_list)

        self._sess.close() #restart session to free memory and to reset profile stats
        tf.reset_default_graph()
        self._sess = tf.Session(config=self._tfconfig) 
         
        sep_net = SeparableNet(scope_idx, self._base_net, self._sess, self._saved_model_path, 
                               self._all_comp_weights_dict, net_desc, self._base_variables)
           
        sep_net.run_performance_analysis(self._blobs_list, self._sess, self._base_outputs_list, 
                                         self._final_layers, self._compression_stats, 
                                         self._base_profile_stats, mAP_base_net, num_imgs_list)
        self._compression_stats.save()
        print(layer_name + ' Kfrac=' + str(Kfrac) + ' complete')
        scope_idx += 1
#         show_all_variables(True, sep_net._net_sep.get_scope())


def pre_tasks():
  return
#   print_for_latex()
#   layers = get_all_compressible_layers()
#   stats2 = CompressionStats('12')
#   stats = CompressionStats('4952_top150')
#   stats = CompressionStats('allLayersKfrac0.5_0.8_0.9_1.0')
#   stats = CompressionStats('allLayersKfrac0.1_0.2_0.3_0.4_0.5_0.8_0.9_1.0')
#   stats = CompressionStats('allLayersKfrac0.1_1.0')
#   stats = CompressionStats('Kfrac0.01-1.0_conv2')
  print(stats)
  
#   stats.merge('12')

  stats = CompressionStats('0.1-1.0_4952')
  stats.plot_correlation_btw_mAP_num_imgs()
  
#   stats.calc_profile_stats_all_nets()
  stats.multivar_regress()
#   stats.save('allLayersKfrac0.8_0.9_1.0')
  
#   stats = CompressionStats('block3_4_mAP_corrn')
#   stats2 = CompressionStats('4952_top150')
#   stats2.print_Kfracs()
#     stats = CompressionStats(filename='CompressionStats_Kfrac0.05-0.6.pi')
#     stats = CompressionStats(filename='CompressionStats_noMap_Kfrac.pi')
#     stats = CompressionStats(filename='CompressionStats_save2.pi')
#     stats.merge('CompressionStats_Kfrac0.32-0.38.pi')
#     stats.merge('CompressionStats_save2.pi')
#   stats.merge('allLayersKfrac0.5')
 
  stats.merge('allLayersKfrac0.9')
#   stats.save('allLayersKfrac0.1_1.0')

#     stats.add_data_type('diff_mean_block3', [0.620057,0.557226,0.426003,0.338981,0.170117,
#                                              0.134585,0.0855217,0.0585074,0.0412037,0.0323449])
#     stats.add_data_type('mAP_4952_top150', [0.0031,0.1165,0.5007,0.6012,0.7630,0.7769,
#                                             0.7831,0.7819,0.7825])
#  
#   stats.save('mergeTest')
#     stats = CompressionStats(filename='CompressionStats_allx5K.pi')
#     stats.plot(plot_type_label=('base_mean','diff_mean','mAP_1000_top150'))

#     stats.plot_single_layers(get_all_compressable_layers(), Kfracs=[0,0.1,0.25,0.5,1.0], 
#                              plot_type_label='diff_mean', ylabel='mean reconstruction error')
#                               plot_type_label='mAP_200_top150', ylabel='mAP')

  types = [['base_mean_bbox_pred', 'base_mean_bbox_pred', False], #1
           ['base_mean_cls_score', 'base_mean_cls_score', False], 
           ['diff_max_bbox_pred', 'diff_max_bbox_pred', False],
           ['diff_max_cls_score', 'diff_max_cls_score', False],
           ['diff_mean_bbox_pred', '$\Delta$ bbox_pred', False], #5
           ['diff_mean_cls_score', '$\Delta$ cls_score', False],
           ['diff_stdev_bbox_pred', 'diff_stdev_bbox_pred', False],
           ['diff_stdev_cls_score', 'diff_stdev_cls_score', False],
           ['flops_count_delta', 'flops_count_delta', False],
           ['flops_frac_delta', '$\Delta$ flops', True], #10
           ['micros_count_delta', 'micros_count_delta', False],
           ['micros_frac_delta', '$\Delta$ micros', True],
           ['param_bytes_count_delta', 'param_bytes_count_delta', False],
           ['param_bytes_frac_delta', '$\Delta$ param bytes', True],
           ['params_count_delta', 'params_count_delta', False], #15
           ['params_frac_delta', '$\Delta$ params', True],
           ['perf_bytes_count_delta', 'perf_bytes_count_delta', False],
           ['perf_bytes_frac_delta', '$\Delta$ perf bytes', True],
           ['output_bytes_frac_delta', '$\Delta$ perf bytes', True],
           ['peak_bytes_frac_delta', '$\Delta$ perf bytes', True], #20
           ['run_count_frac_delta', '$\Delta$ perf bytes', True],
           ['definition_count_frac_delta', '$\Delta$ perf bytes', True],
           ['mAP_100_top150', 'mAP', False],
           ['total_bytes_frac_delta', '$\Delta$ total bytes', True]]
  plot_list = list( types[i-1] for i in [19,20,24,6] )
#   plot_list = list( types[i-1] for i in [10,12,14,18,19,6] )
#   plot_list = list( types[i-1] for i in [14,18,20,19] )
#   stats.plot( plot_list, legend_labels=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'] )
#   exit()
  
#     stats.plot_correlation(['diff_mean'])
  stats.plot_correlation_btw_stats(stats2, 'mAP')
#     stats.plot_correlation(['diff_mean','diff_mean_block3'])
#     stats.plot_correlation(['diff_mean','diff_mean_block3'],[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.75,0.9,1.0])
#     stats.plot_correlation('diff_mean_block3')
#     stats.plot_correlation([0.05,0.1,0.2,0.3,0.32,0.34,0.36,0.38,0.4,0.5,0.6,0.75,0.9,1.0])
  stats.plot_by_Kfracs(#plot_type_label=('mAP_'))
                        plot_type_label=('total_bytes_frac_delta'))
#     stats = CompressionStats(filename='CompressionStats_.pi')
#     print(stats)
#     stats = CompressionStats(filename='CompressionStats_Kfrac0.05-1_noconv1.pi')
    
    
#     stats.plot_K_by_layer(get_all_compressable_layers(), Kfracs = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1], plot_type_label=('mAP_200_top150'))
  stats.plot(('base_mean','diff_mean','var_redux','mAP_10_top100'))
  exit()
   
def calc_reconstruction_errors(base_net, sess, saved_model_path, tfconfig):
#   show_all_variables(show=True)
  
  exp_controller = ExperimentController(base_net, sess, saved_model_path, tfconfig, '16')
  exp_controller.run_exp(num_imgs_list=[5,10,4952])
#   exp_controller.run_exp(num_imgs_list=[5,10,25,50,100,250,500,1000,2000,4952])
#   exp_controller.run_split_net_exp(num_imgs=100)
#   exp_controller.optimise_for_memory(max_iter=50,stats_file_suffix='allLayersKfrac0.1_1.0')
  

    #do the plotting      
#     fig, ax = plt.subplots()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.plot(range(1,Kmax+1),diff_means,'ro-')
#     plt.title('Reconstruction Error - conv1')
#     plt.ylabel('mean abs error')
#     plt.xlabel('K - rank of approximation')
#     plt.show()  
    



