'''
Created on 29 Jul 2017

@author: david
'''
from davelib.resnet_v1_pluri import resnetv1_pluri
from davelib.layer_name import LayerName
from davelib.separable_net import *
from davelib.utils import show_all_variables, stdout_redirector, run_test_metric


class PluripotentNet(SeparableNet):
  
  def __init__(self, base_net, sess, saved_model_path, base_weights_dict, 
               comp_bn_vars_dict, comp_bias_vars_dict, net_desc_pluri, base_variables, 
               filename, timing_results=None):
    SeparableNet.__init__(self, base_net, sess, saved_model_path, base_weights_dict, 
               comp_bn_vars_dict, comp_bias_vars_dict, net_desc_pluri, base_variables, 
               filename, timing_results)

    
  def _init_resnet(self):
      self._net_sep = resnetv1_pluri(batch_size=1, num_layers=101, 
                                     net_desc=self._net_desc, sess=self._sess)

  def _assign_trained_weights_to_unchanged_layers(self):
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      restore_var_dict = {}
 
      for v in self._base_variables:
        name_full = v.op.name
        layer_name = LayerName(name_full, 'net_layer_weights')
#         if layer_name in self._net_desc:
#           continue #skip compressed layers and their BatchNorm params
#         print(name_full + ' ' + layer_name )
        restore_var_dict[name_full] = tf.get_variable(layer_name.layer_weights()) 
 
    saver = tf.train.Saver(restore_var_dict)
    saver.restore(self._sess, self._saved_model_path)
    
  def _assign_trained_weights_to_separable_layers(self):
    all_ops = []
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      for layer_name, Ks in self._net_desc.items():
        source_weights = self._base_weights_dict[layer_name]
        
        for K in Ks:
          layer1_name = LayerName(layer_name +'_sep_K' + str(K) + '/weights','layer_weights')
          layer2_name = LayerName(layer_name +'_K' + str(K) + '/weights','layer_weights')
          dest_weights_1 = tf.get_variable(layer1_name.layer_weights())
          dest_weights_2 = tf.get_variable(layer2_name.layer_weights())
          ops = self._get_assign_ops(source_weights, dest_weights_1, dest_weights_2, K)
          all_ops.extend(ops)
          
          for bn_type in ['beta','gamma','moving_mean','moving_variance']:
            try:
              dest_bn_var_name = layer_name+'_K' + str(K)+'/BatchNorm/'+bn_type
              dest_bn_var = tf.get_variable(dest_bn_var_name)
              source_bn_var_name = layer_name+'/BatchNorm/'+bn_type
              source_bn_var = self._comp_bn_vars_dict[source_bn_var_name]
              assign_op = tf.assign(dest_bn_var, source_bn_var)
              all_ops.append(assign_op)
            except ValueError:
              break #means this compressed layer doesn't have BatchNorm
            
          try:
            dest_bias_var_name = layer_name+'_K' + str(K)+'/biases'
            dest_bias_var = tf.get_variable(dest_bias_var_name)
            source_bias_var_name = layer_name+'/biases'
            source_bias_var = self._comp_bias_vars_dict[source_bias_var_name]
            assign_op = tf.assign(dest_bias_var, source_bias_var)
            all_ops.append(assign_op)
          except ValueError:
            continue #means this compressed layer doesn't have biases
          
      self._sess.run(all_ops)

  def set_active_path_through_net(self, net_desc, sess):
    self._net_sep.set_active_path_through_net(net_desc, sess)

