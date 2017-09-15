'''
Created on 13 Sep 2017

@author: david
'''
import tensorflow as tf
import pickle as pi

from davelib.pluripotent_net import PluripotentNet
from davelib.layer_name import * 
from davelib.compressed_net_description import * 
from davelib.alternate_search import AlternateSearch
from davelib.base_net import BaseNetWrapper
from model.config import cfg
from davelib.compression_stats import CompressionStats
from davelib.optimise_compression import OptimisationResults, plot_results_from_file


class OptimisationManager():
  
  def __init__(self, guide_stats_file_suffix, exp_controller):
    self._exp_controller = exp_controller
    guide_compression_stats = CompressionStats(filename_suffix=guide_stats_file_suffix, 
                                load_from_file=True)
#     guide_compression_stats.remove_redundant_base_profile_stats()
#     guide_compression_stats.save(guide_stats_file_suffix)
    
    mAP_dict = BaseNetWrapper(None).mAP([100])
    
    guide_compression_stats.add_missing_mAP_deltas( mAP_dict[100] )
 
    stats_dict = guide_compression_stats.build_label_layer_K_dict()
    self._efficiency_label = 'float_ops_frac_delta'
#     self._efficiency_label = 'output_bytes_frac_delta'
#     self._efficiency_label = 'param_output_bytes_frac_delta'
#     self._performance_label = 'mAP_10_top150_delta'
#     guide_performance_label = 'mAP_100_top150_delta'
    guide_performance_label = 'diff_mean_cls_score'
    self._performance_label = 'diff_mean_cls_score'
    self._perf_metric_increases_with_degradation = False
#     performance_label = 'diff_mean_cls_prob'
    use_simple = False #if false use the compound greedy serach

    compressed_layers = get_all_compressible_layers()
    compressed_layers = remove_all_conv_1x1(compressed_layers)
    compressed_layers.remove('bbox_pred')
    compressed_layers = remove_layers_not_in_blocks(compressed_layers, [1,2,4])
    print(sorted(compressed_layers))
    comp_layer_label='noconv1x1'
#     compressed_layers = keep_only_conv_not_1x1(compressed_layers)
#     compressed_layers = [LayerName('conv1')]
#     comp_layer_label='conv1'
#     compressed_layers = [LayerName('block4/unit_3/bottleneck_v1/conv2')]
#     comp_layer_label='b4_u3_conv2'
    Kfracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    Kfracs_label='0.1-1.0'
#     Kfracs = [0.6]
#     Kfracs_label='0.6'
#     self._num_imgs_list=[10]
    self._num_imgs_list=[]
#     self._output_layers = []
    self._output_layers = self._exp_controller._final_layers

    filename = 'pluri_net_' + Kfracs_label + '_' + comp_layer_label

    self._pluri_net, self._mAP_base_net = exp_controller.init_plurinet(self._num_imgs_list,
                                                                 compressed_layers, Kfracs, 
                                                                 filename)
    
    self._efficiency_dict = stats_dict[self._efficiency_label] #layer->K
    self._performance_dict = stats_dict[guide_performance_label]
    
    efficiency_dict_keys = remove_all_conv_1x1(self._efficiency_dict.keys())
    efficiency_dict_keys.remove('bbox_pred') #must remove or it will always 'win' as has no effect on cls_score
    efficiency_dict_keys = remove_layers_not_in_blocks(efficiency_dict_keys, [1,2,4])
    efficiency_dict = { k: self._efficiency_dict[k] for k in efficiency_dict_keys }
#     efficiency_dict = remove_all_except_conv2_first_conv1(efficiency_dict)


    def remove_positive_delta_compressions(efficiency_dict):
      new_dict = {}
      removed_layers = []
      for k, d in efficiency_dict.items(): 
        d2 = {K: val for K, val in d.items() if val<0}
        if len(d2) > 0:
          new_dict[k] = d2
        else:
          removed_layers.append(k)
          print(k + ' has no +ve delta compressions')
      return new_dict, removed_layers

    self._efficiency_dict, removed_layers = remove_positive_delta_compressions(efficiency_dict)
    compressed_layers = list(set(compressed_layers)-set(removed_layers))
    
    Kfracs = [0.9]
    self._initial_net_desc = build_net_desc(Kfracs, compressed_layers)
    
    self._search_algo = AlternateSearch(self._initial_net_desc, 
                                        self._efficiency_dict, 
                                        self._performance_dict, 
                                        self._perf_metric_increases_with_degradation,
                                        compressed_layers)

  def _expected_delta(self, compression_step, metric_dict): 
    layer, K_old, K_new = compression_step._layer, compression_step._K_old, compression_step._K_new  
    new_metric = 0
    if K_new != UNCOMPRESSED:
      new_metric = metric_dict[layer][K_new]
    old_metric = 0
    if K_old != UNCOMPRESSED:
      old_metric = metric_dict[layer][K_old]
    return new_metric - old_metric

  def _get_efficiency_metric(self, net_desc):
    sum_ = 0.0
    for layer, K in net_desc.items():
      sum_ += self._efficiency_dict[layer][K]
    return sum_

  def run_search(self, max_iter):
    compression_stats = CompressionStats(load_from_file=False)
    old_performance_metric = 0
    cum_efficiency_delta = 0 
    opt_results = [] 

    for itern in range(max_iter):#each iteration compresses 1 layer (a bit more)
      print(str(itern+1), end=' ')
      
#       if itern == 0:
#         layer_K_dict = self._initial_net_desc
#         compression_step = None
#       else:
      net_desc, compression_step = self._search_algo.get_next_model()
      
      expected_efficiency_delta = self._expected_delta( compression_step, self._efficiency_dict )   
      expected_perf_delta = self._expected_delta( compression_step, self._performance_dict)   
#       net_desc = CompressedNetDescription(layer_K_dict)
      
      self._pluri_net.set_active_path_through_net(net_desc, self._exp_controller._sess)
      
      self._pluri_net.run_performance_analysis(net_desc, 
                                               self._exp_controller._blobs_list, 
                                               self._exp_controller._sess, 
                                               self._exp_controller._base_outputs_list, 
                                               self._output_layers, 
                                               compression_stats, 
                                               self._exp_controller._base_profile_stats, 
                                               self._mAP_base_net,
                                               self._num_imgs_list, 
                                               run_profile_stats=False)

      new_efficiency_metric = self._get_efficiency_metric(net_desc)
      new_performance_metric = compression_stats.get(net_desc, self._performance_label)
    
      actual_perf_delta = new_performance_metric - old_performance_metric
      print('\t%s=%f'%(self._performance_label,new_performance_metric))
      
      this_iter_res = OptimisationResults(expected_efficiency_delta, actual_perf_delta,
                        expected_perf_delta, net_desc, compression_step, 
                        self._performance_label, self._efficiency_label, 
                        new_efficiency_metric, new_performance_metric)
      opt_results.append( this_iter_res )
      self._search_algo.set_model_results(this_iter_res)

      cum_efficiency_delta += expected_efficiency_delta
      print('\t%s=%f'%(self._efficiency_label,cum_efficiency_delta))
      if expected_perf_delta:
        print('\texpected/actual \u0394perf=%.3f / %.3f'%(expected_perf_delta,actual_perf_delta))
      old_performance_metric = new_performance_metric
      
      pi.dump( opt_results, open( 'opt_results_output_bytes_clsscore', "wb" ) )

    print('Final optimised net description:\n' + str(net_desc))
    
    type_labels = ['float_ops_frac_delta','float_ops_count_delta',
                   'output_bytes_frac_delta','output_bytes_count_delta',
                   'parameters_frac_delta','parameters_count_delta']
#     compression_stats.plot_data_type_by_Kfracs(type_labels)
