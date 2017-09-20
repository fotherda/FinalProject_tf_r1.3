'''
Created on 13 Sep 2017

@author: david
'''

from davelib.optimise_compression import OptimisationResults, plot_results_from_file, \
                    scatter_plot_results_from_file, plot_compression_profile_from_file

from davelib.layer_name import *

def display():

  compressed_layers = get_all_compressible_layers()
  compressed_layers = remove_all_conv_1x1(compressed_layers)
  compressed_layers.remove('bbox_pred')
  Kfrac_init = 0.9



#   scatter_plot_results_from_file('alt_srch_res_1000_flops_Kfrac0.9')
#   plot_compression_profile_from_file('opt_results_output_bytes_clsscore')
#   scatter_plot_results_from_file('opt_results_output_bytes_clsscore')
#   plot_compression_profile_from_file('alt_srch_res_0_6_allLayers_effic_perf', compressed_layers, Kfrac_init=Kfrac_init)
#   plot_compression_profile_from_file('alt_srch_res_0.9_allLayers', compressed_layers, Kfrac_init=Kfrac_init)
#   plot_results_from_file('alt_srch_res_comp')
  plot_results_from_file('alt_srch_res_0_9_perf')
#   plot_results_from_file('alt_srch_res_0_6_allLayers_effic_perf')
#   plot_results_from_file('alt_srch_res')
#   plot_results_from_file('alt_srch_res_flops_Kfrac0.9_allLayers')
#   plot_results_from_file('alt_srch_res_1000_flops_Kfrac0.9')
#   plot_results_from_file('opt_results')
#   plot_results_from_file('opt_path_flops_eff_perf')
#   plot_results_from_file('opt_path_output_bytes_mAP_simple_10')
#   plot_results_from_file('opt_path_output_bytes_eff_perf')
#   plot_results_from_file('opt_path_output_bytes_eff_perf_10')
#   plot_results_from_file('opt_results_output_bytes_clsscore')
  exit()

#   print_for_latex()
#   layers = get_all_compressible_layers()
#   stats = CompressionStats('4952_top150')
#   stats = CompressionStats('allLayersKfrac0.5_0.8_0.9_1.0')
#   stats = CompressionStats('allLayersKfrac0.1_0.2_0.3_0.4_0.5_0.8_0.9_1.0')
#   stats = CompressionStats('allLayersKfrac0.1_1.0')
#   stats = CompressionStats('Kfrac0.01-1.0_conv2')

  #Kfrac_mAP_X plots
#   stats = CompressionStats('0.1-1.0_4952')
#   stats.merge('0.32-0.38')
#   stats.plot_by_Kfracs(plot_type_label='mAP_5_top150')


  #mAP_corrn plots
#   stats = CompressionStats('0.1-1.0_4952')
#   stats.merge('0.32-0.38')
#   stats.plot_correlation_btw_mAP_num_imgs(min_mAP=0.0)
  
  
  
  
  print(stats)
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
 
#   stats.merge('allLayersKfrac0.9')
#   stats.save('allLayersKfrac0.1_1.0')

#     stats.add_data_type('diff_mean_block3', [0.620057,0.557226,0.426003,0.338981,0.170117,
#                                              0.134585,0.0855217,0.0585074,0.0412037,0.0323449])
#     stats.add_data_type('mAP_4952_top150', [0.0031,0.1165,0.5007,0.6012,0.7630,0.7769,
#                                             0.7831,0.7819,0.7825])
#  
#   stats.save('mergeTest')
#     stats = CompressionStats(filename='CompressionStats_allx5K.pi')
#     stats.plot(plot_type_label=('base_mean','diff_mean','mAP_1000_top150'))

#   stats.plot_single_layers(get_all_compressable_layers(), Kfracs=[0,0.1,0.25,0.5,1.0], 
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
    
    
#             # to produce flops_cpu_params_profile.png graphs: 
#             stats = CompressionStats('0.1-1.0')
#             type_labels = [
#                           'float_ops_frac_delta','float_ops_count_delta',
#                           'cpu_exec_micros_frac_delta','cpu_exec_micros_count_delta',
#                            'parameters_frac_delta','parameters_count_delta'
#           #                 'output_bytes_frac_delta','output_bytes_count_delta',
#           #                 'residual_bytes_frac_delta','residual_bytes_count_delta',
#           #                 'peak_bytes_frac_delta','peak_bytes_count_delta',
#           #                 'requested_bytes_frac_delta','requested_bytes_count_delta',
#                           ]
#             stats.plot_data_type_by_Kfracs(type_labels)

    
    
#     stats.plot_K_by_layer(get_all_compressable_layers(), Kfracs = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1], plot_type_label=('mAP_200_top150'))
  stats.plot(('base_mean','diff_mean','var_redux','mAP_10_top100'))
  exit()





    #do the plotting      
#     fig, ax = plt.subplots()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.plot(range(1,Kmax+1),diff_means,'ro-')
#     plt.title('Reconstruction Error - conv1')
#     plt.ylabel('mean abs error')
#     plt.xlabel('K - rank of approximation')
#     plt.show()  
    
