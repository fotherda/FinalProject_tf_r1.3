'''
Created on 29 Jul 2017

@author: david
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os.path

from model.config import cfg
from davelib.resnet_v1_sep import resnetv1_sep
from davelib.layer_name import LayerName
from tensorflow.python.framework import ops
from collections import OrderedDict
from davelib.profile_stats import ProfileStats
from davelib.utils import show_all_variables, stdout_redirector, run_test_metric
from utils.timer import Timer


def mismatch_count(base_outputs, sep_outputs):
  count = 0
  for i in range(base_outputs.shape[0]):
    base_output = base_outputs[i]
    sep_output = sep_outputs[i]
    max_idx_base = np.argmax(base_output)
    max_idx_sep = np.argmax(sep_output)
    if max_idx_base != max_idx_sep:
      count += 1
  return count    
  
def check_reconstruction_error(V, H, W, K):
  s = W.shape
  assert s[0] == s[1]
  C = s[2]
  d = s[0]
  N = s[3]
  
  f_norm_sum = 0
  plots = np.zeros((2,N,d,d))
  for n in range(N):
    for c in range(C):
      
      sum_prod = np.zeros((d,d))
      for k in range(K):
        p = np.outer(V[c,k,:], H[n,k,:])
        sum_prod += p
      diff = np.subtract(W[:,:,c,n], sum_prod)
      f_norm = np.linalg.norm(diff, ord='fro')
      f_norm_sum += f_norm
      plots[0,n,:,:] = sum_prod
      plots[1,n,:,:] = W[:,:,c,n]
  
  return f_norm_sum, plots

def plot_filters(plots_dict, n):
  fig = plt.figure(figsize=(15,8))
  n_columns = n + 1
  n_rows = len(plots_dict) + 1
  
  a = plt.subplot(n_rows, n_columns, 1)
  a.text(0.75, 0.5, 'Base Model\nFilter', fontsize=16, horizontalalignment='center', verticalalignment='center')
  imgs = plots_dict[ list(plots_dict.keys())[0] ]
  for i in range(n):
    plt.subplot(n_rows, n_columns, i+2)
    plt.title('channel ' + str(i+1), fontsize=16)
    plt.imshow(imgs[1,i,:,:], interpolation="nearest", cmap="gray")
  
  for j, (k, imgs) in enumerate(plots_dict.items()):
    a = plt.subplot(n_rows, n_columns, (j+1)*n_columns + 1)
    a.text(0.75, 0.5, 'Low-rank\nFilter\nK=' + str(k), fontsize=16,horizontalalignment='center', verticalalignment='center')
    for i in range(n):
      plt.subplot(n_rows, n_columns, (j+1)*n_columns+i+2)
      plt.imshow(imgs[0,i,:,:], interpolation="nearest", cmap="gray")

  axes = fig.get_axes()
  for ax in axes:
    ax.axis('off')
  plt.tight_layout()
  plt.show()


class SeparableNet(object):
  
  def __init__(self, base_net, sess, saved_model_path, base_weights_dict, 
               comp_bn_vars_dict, comp_bias_vars_dict, net_desc, base_variables, filename=None):

    self._base_net = base_net
    self._base_weights_dict = base_weights_dict
    self._comp_bn_vars_dict = comp_bn_vars_dict
    self._comp_bias_vars_dict = comp_bias_vars_dict
    self._net_desc = net_desc
    self._sess = sess
    self._saved_model_path = saved_model_path
    self._base_variables = base_variables
    self._init_resnet()
    self._net_sep.create_architecture(self._sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    
    restore = False
    if filename:
      save_dir = saved_model_path[:saved_model_path.rfind('/')+1]
      save_file = save_dir + filename
      restore_file = save_file + '.meta'
      if os.path.isfile(restore_file):
        restore = True

    if restore:
      _t = Timer()
      _t.tic()
      saver = tf.train.Saver()
      saver.restore(sess,tf.train.latest_checkpoint(save_dir))
      _t.toc()
#       show_all_variables(True, self._net_sep.get_scope())
      print('Net restore from file took: {:.3f}s' .format( _t.diff))
    else:    
      self._assign_weights()

    if filename and not restore: #save this model with it's variables and weights
      saver = tf.train.Saver()
      saver.save(sess, save_file)

  def _init_resnet(self):
    self._net_sep = resnetv1_sep(batch_size=1, num_layers=101, 
                        base_weights_dict=self._base_weights_dict, net_desc=self._net_desc)
    
  def _assign_weights(self):
    _t = Timer()
    _t.tic()
#     self._net_sep = resnetv1_sep(scope_idx, batch_size=1, num_layers=101, 
#                         base_weights_dict=self._base_weights_dict, net_desc=self._net_desc)
#     self._net_sep.create_architecture(self._sess, "TEST", 21,
#                           tag='default', anchor_scales=[8, 16, 32])
#     show_all_variables(True, self._net_sep.get_scope())

    self.assign_trained_weights_to_unchanged_layers()
    self.assign_trained_weights_to_separable_layers()  
    _t.toc()
    print('PluripotentNet init took: {:.3f}s' .format( _t.diff))

    
  def _assign_trained_weights_to_unchanged_layers(self):
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      restore_var_dict = {}

      for v in self._base_variables:
        name_full = v.op.name
        layer_name = LayerName(name_full, 'net_layer_weights')
        if layer_name in self._net_desc:
          continue
#         print(name_full + ' ' + layer_name )
        restore_var_dict[name_full] = tf.get_variable(layer_name.layer_weights()) 

    saver = tf.train.Saver(restore_var_dict)
    saver.restore(self._sess, self._saved_model_path)
    
  
  def _assign_trained_weights_to_separable_layers(self):
    all_ops = []
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      for layer_name in self._net_desc:
#       for layer_name, source_weights in self._comp_weights_dict.items():
        source_weights = self._base_weights_dict[layer_name]
        layer1_name = LayerName(layer_name +'_sep/weights','layer_weights')
        dest_weights_1 = tf.get_variable(layer1_name.layer_weights())
        dest_weights_2 = tf.get_variable(layer_name.layer_weights())
        K = (self._net_desc[layer_name])[0]
        ops = self._get_assign_ops(source_weights, dest_weights_1, dest_weights_2, K)
        all_ops.extend(ops)
          
      self._sess.run(all_ops)

  def _get_assign_ops(self, source_weights, dest_weights_1, dest_weights_2, K, plot=False):
    if plot :
      f_norms = np.empty(K+1)
      plots_dict = OrderedDict()
      Ks = [21,5,2,1]
      for k in Ks:
        V, H = self._get_low_rank_filters(source_weights, k)
        f_norms[k], plots = check_reconstruction_error(V, H, source_weights, k)
        plots_dict[k] = plots
      plot_filters(plots_dict, 4)
    else:
      if len(source_weights.shape) == 4: #convolutional layer
        H,W,C,N = tuple(source_weights.shape)
        if H==1 and W==1: #1x1 conv layer
          M1, M2 = self._get_low_rank_1x1_filters(source_weights, K)
        else:
          M1, M2 = self._get_low_rank_filters(source_weights, K)
          M1 = np.moveaxis(M1, source=2, destination=0)
          M1 = np.expand_dims(M1, axis=1)
          M2 = np.swapaxes(M2, axis1=2, axis2=0)
          M2 = np.expand_dims(M2, axis=0)
      elif len(source_weights.shape) == 2: #fc layer
        M1, M2 = self._get_low_rank_weights_for_fc_layer(source_weights, K)
        
    assign_op_1 = tf.assign(dest_weights_1, M1)
    assign_op_2 = tf.assign(dest_weights_2, M2)
    
    return [assign_op_1, assign_op_2]

  def _trim_outputs(self, base_output, sep_output):
    if base_output.shape != sep_output.shape:
      s = sep_output.shape
      base_output = base_output[:s[0],:]
    return base_output

  def run_performance_analysis(self, blobs_list, sess, base_outputs_list, output_layers, 
                               compression_stats, base_profile_stats, mAP_base_net=None, 
                               num_imgs_list=[], plot=False, run_profile_stats=True):

    if base_outputs_list is None:
      base_outputs_list = self._base_net.get_outputs_multi_image(blobs_list, output_layers, sess)
    sep_outputs_list, run_metadata_list = self._net_sep.get_outputs_multi_image(blobs_list, output_layers, sess)

    for name in output_layers: # probably cls_score and bbox_pred - the 2 final layers
      diffs_cat = None
      base_output_list = []
      mismatch_cnt = 0
      for base_outputs, sep_outputs in zip(base_outputs_list, sep_outputs_list): #loop once per test img
        base_output = base_outputs[name]
        sep_output = sep_outputs[name]
      
        if plot:
          self.plot_outputs(base_output, sep_output, name)
        
#         mismatch_cnt += mismatch_count(base_output, sep_output)
        
        base_output_trimmed = self._trim_outputs(base_output, sep_output)
        diff = np.subtract(base_output_trimmed, sep_output)
        if diffs_cat is not None:
          diffs_cat = np.append(diffs_cat, diff)
        else:
          diffs_cat = diff
          
        base_output_list.append(base_output)
        
      base_output_mean = np.mean(np.absolute(base_output_list)) 
      diff_mean_abs = np.mean(np.absolute(diffs_cat))
      diff_stdev_abs = np.std(np.absolute(diffs_cat))
      diff_max_abs = np.max(np.absolute(diffs_cat))
    
#       print('base mean=', base_output_mean, ' diff mean=', diff_mean_abs, 
#             ' stdev=', diff_stdev_abs, ' max=', diff_max_abs)

      compression_stats.set(self._net_desc, 'base_mean_'+name, base_output_mean)
      compression_stats.set(self._net_desc, 'diff_mean_'+name, diff_mean_abs)
      compression_stats.set(self._net_desc, 'diff_stdev_'+name, diff_stdev_abs)
      compression_stats.set(self._net_desc, 'diff_max_'+name, diff_max_abs)
#       compression_stats.set(self._net_desc, 'mismatch_count_'+name, mismatch_cnt)

#         num_imgs = 4952
    if len(num_imgs_list) > 0:
      suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      det_filename = "_".join(['default/res101_comp_net', suffix]) # e.g. 'mylogfile_120508_171442'
      mAP_dict = run_test_metric(num_imgs_list, self._net_sep, sess, filename=det_filename)
      compression_stats.set(self._net_desc, 'detections_file', det_filename)
      for num_imgs, mAP in mAP_dict.items():
        compression_stats.set(self._net_desc, 'mAP_%d_top%d'%(num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N), mAP)
        compression_stats.set(self._net_desc, 'mAP_%d_top%d_delta'%
                              (num_imgs,cfg.TEST.RPN_POST_NMS_TOP_N), mAP - mAP_base_net)
        print('mAP=%f, diff_mean_abs=%f'%(mAP,diff_mean_abs))
    else:
      print('diff_mean_abs=%f'%diff_mean_abs)
      
    if run_profile_stats:
      profile_stats = ProfileStats(run_metadata_list, tf.get_default_graph())
      compression_stats.set_profile_stats(self._net_desc, profile_stats, base_profile_stats)
  
  def run_inference(self, blobs, base_outputs, compressed_layers):
  
    outputs = self._net_sep.test_image_2(self._sess, blobs['data'], blobs['im_info'], 
                                         compressed_layers)
    
    for layer_name in compressed_layers:
      base_output = base_outputs[layer_name.net_layer(self._base_net.get_scope())]
      sep_output = outputs[layer_name.net_layer(self._net_sep.get_scope())]
      
      self._plot_outputs(base_output, sep_output)
      diff = np.subtract(base_output, sep_output)
      
      base_output_mean = np.mean(np.absolute(base_output)), 
      diff_mean_abs = np.mean(np.absolute(diff))
      diff_stdev_abs = np.std(np.absolute(diff)),
      diff_max_abs = np.max(np.absolute(diff))
      
      print('conv1 mean=', base_output_mean, ' mean=', diff_mean_abs, 
            ' stdev=', diff_stdev_abs, ' max=', diff_max_abs)
    
    return base_output_mean, diff_mean_abs, diff_stdev_abs, diff_max_abs
    
  def _tensor_to_matrix(self, W_arr):
    # convert 4-D tensor to 2-D using bijection from Tai et al 2016
    s = W_arr.shape
    assert s[0] == s[1]
    C = s[2]
    d = s[0]
    N = s[3]
    W = np.empty([C*d, d*N])
    
    for i1 in range(1,C+1):
      for i2 in range(1,d+1):
        for i3 in range(1,d+1):
          for i4 in range(1,N+1):
            j1 = (i1-1)*d + i2
            j2 = (i4-1)*d + i3
            W[j1-1, j2-1] = W_arr[i2-1,i3-1,i1-1,i4-1] #subtract 1 to adjust for zero based arrays
  
    return C,d,N,W  
    
  def _get_low_rank_1x1_filters(self, weights, K): #for convolutional layers
    W = np.squeeze(weights)
    U,Dvec,Qt = np.linalg.svd(W) # U=C x C, Q=N x N, D=min(C,N), 
    D = np.diagflat(Dvec)
    P = np.dot(D[:K,:K], Qt[:K,:] )
    P = np.expand_dims(P, 0)
    P = np.expand_dims(P, 0)
    U = U[:,:K]
    U = np.expand_dims(U, 0)
    U = np.expand_dims(U, 0)
                      
    return U, P

  def _get_low_rank_filters(self, weights, K): #for convolutional layers
    C,d,N,W = self.tensor_to_matrix(weights) # W=Cd x Nd
    U,D,Qt = np.linalg.svd(W) # U=Cd x Cd, Q=Nd x Nd, D=Cd, 
    Q = np.transpose(Qt)
    
    V = np.empty([C,K,d])
    H = np.empty([N,K,d])
    
    for k in range(K):
      for j in range(d):
        for c in range(C):
          V[c,k,j] = U[c*d + j, k] * np.sqrt(D[k])
        for n in range(N):
          H[n,k,j] = Q[n*d + j, k] * np.sqrt(D[k])
  
    return V, H 

  def _get_low_rank_weights_for_fc_layer(self, weights, K):
    U,Dvec,Qt = np.linalg.svd(weights) # U=C x C, Q=N x N, D=min(C,N), 
    D = np.diagflat(Dvec)
    return U[:,:K], np.dot(D[:K,:K], Qt[:K,:])
  
  def _plot_outputs(self, units, units_sep, name):
      filters = 4
      fig = plt.figure(figsize=(15,8))
      n_columns = filters+1
      n_rows = 1
      
      if name in self._net_desc.keys():
        K = self._net_desc[name]
      else:
        K = 0
      
#       a = plt.subplot(n_rows, n_columns, 1)
#       a.text(0.75, 0.5, 'Base\nModel', fontsize=16, horizontalalignment='center', verticalalignment='center')
      a = plt.subplot(n_rows, n_columns, 1)
#       a = plt.subplot(n_rows, n_columns, n_columns + 1)
      a.text(0.75, 0.5, 'Compressed\nModel\nK=' + str(K), fontsize=16,horizontalalignment='center', verticalalignment='center')
#       a = plt.subplot(n_rows, n_columns, n_columns*2 + 1)
#       a.text(0.5, 0.5, 'Reconstruction\nError', fontsize=16,horizontalalignment='center', verticalalignment='center')
      
      for i in range(filters):
          combined_data = np.array([units[0,:,:,i],units_sep[0,:,:,i]])
          _min, _max = np.amin(combined_data), np.amax(combined_data)

#           a = plt.subplot(n_rows, n_columns, i+2)
#           plt.title('Channel ' + str(i+1), fontsize=16)
#           plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
          
          a = plt.subplot(n_rows, n_columns, i+2)
#           a = plt.subplot(n_rows, n_columns, n_columns + i+2)
          plt.imshow(units_sep[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
#           a = plt.subplot(n_rows, n_columns, n_columns*2 + i+2)
#           diff = np.subtract(units[0,:,:,i], units_sep[0,:,:,i])
#           plt.imshow(diff, interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
      axes = fig.get_axes()
      for ax in axes:
        ax.axis('off')
#       fig.suptitle('Layer: ' + name, fontsize=18, x=0.5, y=0.02, horizontalalignment='center', verticalalignment='center')
      plt.tight_layout()
      plt.show()
#       plt.savefig(figure_path+'/output_imgs_K'+str(self._K)+'.png')

