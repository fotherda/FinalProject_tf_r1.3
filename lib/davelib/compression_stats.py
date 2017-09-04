'''
Created on 13 Jul 2017

@author: david
'''
import numpy as np
import pickle as pi
import matplotlib.pyplot as plt
import re, time
import sys  

# reload(sys)  
# sys.setdefaultencoding('utf8')

from collections import OrderedDict,defaultdict
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter
from sklearn import linear_model
from kernel_regression import KernelRegression
from sklearn.svm import SVR
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV, learning_curve
# from sklearn.learning_curve import learning_curve

from functools import total_ordering
from mpl_toolkits.mplot3d import *
from random import random, seed
from matplotlib import cm

root_filename = 'CompressionStats_'


@total_ordering
class CompressedNetDescription(dict):
   
  def __init__(self, Ks_by_layer_dict):
    items = []
#     self._Ks = []
    for layer, Ks in Ks_by_layer_dict.items():
      self[layer] = Ks
#       self._Ks.append(K)
      items.extend( (layer, tuple(Ks)) )
    self._key = tuple(items)

#   def __init__(self, compressed_layers, Ks):
#     items = []
# #     self._Ks = []
#     for layer, K in zip(compressed_layers,Ks):
#       self[layer] = K
# #       self._Ks.append(K)
#       items.extend((layer, K))
#     self._key = tuple(items)
      
#   def __key(self):
#     return self['conv1']
  
  def __key(self):
    return self._key
   
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      if not self.__key() and not other.__key():
        return True
      else:
        return self.__key() == other.__key()
    return False
  
  def __hash__(self):
    return hash(self.__key())
  
  def __lt__(self, other):
    if not self:
      if not other:
        return True
      else:
        return True
    elif not other:
      return False
    else:
      return list(self.keys())[0] < list(other.keys())[0]
#     return list(self.keys())[0] < list(other.keys())[0]
  
#   def __str__(self):
#     return '\n'.join(map(str, sorted(self.items())))

  def get_Kfrac(self):
    if len(self) == 0: # uncompressed
      return 0.0
    for layer, K in self.items():
      if 'block4' in layer and 'conv2' in layer:
        Kfrac = K / 768.0
        break
      elif 'block3' in layer and 'conv2' in layer:
        Kfrac = K / 384.0
        break
    return Kfrac

class CompressionStats(object):

  def __init__(self, filename_suffix='', load_from_file=True, all_Kmaxs_dict=None):
    self._filename_suffix = filename_suffix
    self._all_Kmaxs_dict = all_Kmaxs_dict
    if load_from_file: #load from pickle file
      self.load_from_file(root_filename+filename_suffix+'.pi')
    else:  
      self._stats = defaultdict( dict )
      
  def __str__(self):
    str_list = []
    for net_desc, d in sorted(self._stats.items()):
      str_list.append(str(net_desc))
      str_list.append('\n')
      
      for type_label, value in sorted(d.items()):
        str_list.append(type_label + '\t' + str(value) + '\n')
    return ''.join(str_list)
  
  
  def print_Kfracs(self):
    str_list = []
    str_list.append('Kfracs:\n')
    for net_desc, d in self._stats.items():
      str_list.append('%.2f'%net_desc.get_Kfrac())
      str_list.append('\n')

    str_list.append('Data Types:\n')
    for type_label in d:
        str_list.append(type_label + '\n')
    
    print( ''.join(str_list) )
  
  
  def load_from_file(self, filename):  
    self._stats = pi.load( open( filename, "rb") ) 
#     self._stats = pi.load( open( filename, "rb"), encoding='latin1' ) 
#     print self._stats   

  def set(self, net_desc, type_label, value):
    self._stats[net_desc][type_label] = value

  def get(self, net_desc, type_label):
    return self._stats[net_desc][type_label]

  def set_profile_stats(self, net_desc, profile_stats, base_profile_stats):
    self.set(net_desc, 'profile_stats', profile_stats)
    self.set(net_desc, 'base_profile_stats', base_profile_stats)
    self._set_profile_stats(net_desc, profile_stats, base_profile_stats, 'count_delta')
    self._set_profile_stats(net_desc, profile_stats, base_profile_stats, 'frac_delta')
  
  def _set_profile_stats(self, net_desc, profile_stats, base_profile_stats, count_or_frac):
    func = getattr(profile_stats, count_or_frac)
    
    self.set(net_desc, 'params_'+count_or_frac, func(
                          base_profile_stats, '_param_stats', 'total_parameters'))
    self.set(net_desc, 'flops_'+count_or_frac, func(
                          base_profile_stats, '_perf_stats', 'total_float_ops'))
    self.set(net_desc, 'param_bytes_'+count_or_frac, func(
                          base_profile_stats, '_param_stats', 'total_requested_bytes'))
    self.set(net_desc, 'perf_bytes_'+count_or_frac, func(
                          base_profile_stats, '_perf_stats', 'total_requested_bytes'))
    self.set(net_desc, 'micros_'+count_or_frac, func(
                          base_profile_stats, '_perf_stats', 'total_cpu_exec_micros'))
    self.set(net_desc, 'peak_bytes_'+count_or_frac, func(
                          base_profile_stats, '_scope_all_stats', 'total_peak_bytes'))
    self.set(net_desc, 'output_bytes_'+count_or_frac, func(
                          base_profile_stats, '_scope_all_stats', 'total_output_bytes'))
    self.set(net_desc, 'run_count_'+count_or_frac, func(
                          base_profile_stats, '_scope_all_stats', 'total_run_count'))
    self.set(net_desc, 'definition_count_'+count_or_frac, func(
                          base_profile_stats, '_scope_all_stats', 'total_definition_count'))
    
    func = getattr(profile_stats, 'total_bytes_'+count_or_frac)
    self.set(net_desc, 'total_bytes_'+count_or_frac, func(base_profile_stats))
    
  def save(self, suffix=None):
    if not suffix:
      suffix = self._filename_suffix
    pi.dump( self._stats, open( '%s%s.pi'%(root_filename,suffix), "wb" ) )


  def add_data_type(self, type_label, values): #values must be in order to match sorted net descriptions
    for i, (K_by_layer_dict, d) in enumerate(sorted(self._stats.items())):
      print(str(K_by_layer_dict['conv1']))
      d[type_label] = values[i]
    
  def merge(self, filename_suffix):
    filename = root_filename+filename_suffix+'.pi'
    other_stats = pi.load( open( filename, "rb" ) ) 
    
    for net_desc, data_dict in sorted(other_stats.items()):
      if net_desc in self._stats:
        print('net_desc already exists')
        for type_label, d in sorted(data_dict.items()):
          if type_label not in self._stats[net_desc]:
            self._stats[net_desc][type_label] = d
            print('added ' + type_label) 
      else:
        self._stats[net_desc] = data_dict


  def calc_profile_stats_all_nets(self):
    for net_desc in sorted(self._stats):
      profile_stats = self._stats[net_desc]['profile_stats']
      base_profile_stats = self._stats[net_desc]['base_profile_stats']
      self._set_profile_stats(net_desc, profile_stats, base_profile_stats, 'count_delta')
      self._set_profile_stats(net_desc, profile_stats, base_profile_stats, 'frac_delta')

  #only works when each net_desc has 1 compressed layer
  def build_label_layer_K_dict(self):
    new_dict = defaultdict( lambda: defaultdict (lambda: defaultdict(float)) )
    for net_desc, d in self._stats.items():
#       if len(net_desc) != 1:
#         continue
#       layer = list(net_desc.keys())[0]
#       K = net_desc[layer]
      for layer, K in net_desc.items():
        for type_label, value in d.items():
          new_dict[type_label][layer][K] = value
    return new_dict
  

  
  def multivar_regress(self):
#     X, y = self.regression_data()
    X, y = self.regression_data_split()
    X = np.array(X)
    y = np.array(y)
    
    pb = X[:,0].argsort()
    Xb = X[pb]
    yb = y[pb]

    X1 = np.delete(X, 1, 1)
    p1 = X1[:,0].argsort()
    X1 = X1[p1]
    y1 = y[p1]
    
    X2 = np.delete(X, 0, 1)
    p2 = X2[:,0].argsort()
    X2 = X2[p2]
    y2 = y[p2]
    
    x_range=np.arange(0, 0.025, 0.001)                # generate a mesh
    y_range=np.arange(0, 1.3, 0.02)
    x_surf, y_surf = np.meshgrid(x_range, y_range)
    Xpred = np.stack((x_surf.flatten(), y_surf.flatten()), axis=1)

    svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e-1, 1e0, 1e1, 1e2],
                               "gamma": np.logspace(-2, 2, 10)})
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    t0 = time.time()
    y_svrb = svr.fit(Xb, yb).predict(Xpred)
    print("SVR complexity and bandwidth selected and model fitted in %.3f s" % (time.time() - t0))

    score_svr = svr.score(Xb, yb)
    y_svr1 = svr.fit(X1, y1).predict( np.expand_dims(x_range,1) )
    score_svr1 = svr.score(X1, y1)
    y_svr2 = svr.fit(X2, y2).predict( np.expand_dims(y_range,1) )
    score_svr2 = svr.score(X2, y2)
    
    t0 = time.time()
    y_krb = kr.fit(Xb, yb).predict(Xpred)
    print("KR including bandwith fitted in %.3f s" % (time.time() - t0))
    
    score_kr = kr.score(Xb, yb)
    y_kr1 = kr.fit(X1, y1).predict( np.expand_dims(x_range,1) )
    score_kr1 = kr.score(X1, y1)
    y_kr2 = kr.fit(X2, y2).predict( np.expand_dims(y_range,1) )
    score_kr2 = kr.score(X2, y2)

    print('R^2 / coeff determination:')
    print('  SVR model: cls_score=%0.3f bbox_pred=%0.3f both=%0.3f' % (score_svr1, score_svr2, score_svr))
    print('  KR model: cls_score=%0.3f bbox_pred=%0.3f both=%0.3f' % (score_kr1, score_kr2, score_kr))
    
#     R^2 / coeff determination:
#   SVR model: cls_score=0.675 bbox_pred=0.518 both=0.512
#   KR model: cls_score=0.848 bbox_pred=0.320 both=0.881

    
    
    ###############################################################################
    # Visualize models
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')               # to work in 3d
#     
#     z_surf = np.reshape(y_krb, x_surf.shape)          
#     surf = ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.coolwarm, alpha=0.5, rstride=1, cstride=1);    # plot a 3d surface plot
#     fig.colorbar(surf, shrink=0.5, aspect=5)
# 
#     ax.scatter(X[:,0], X[:,1], y, s=1, c='k')                        # plot a 3d scatter plot
#     
#     ax.set_xlabel('cls_score', fontsize=16)
#     ax.set_ylabel('bbox_pred', fontsize=16)
#     ax.set_zlabel('mAP', fontsize=16)
#     plt.show()
    
    fig = plt.figure()
    plt.scatter(X1[:,0], y1, c='k', s=1, label='data')
#     plt.plot(x_range, y_kr1, c='g', label='Kernel Regression')
#     plt.plot(x_range, y_svr1, c='r', label='SVR')
    plt.xlabel('cls_score')
    plt.ylabel('mAP')
    plt.ylim(0, 0.85)
#     plt.title('Classification score difference as proxy for model performance/')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.scatter(X2[:,0], y2, c='k', s=1, label='data')
#     plt.plot(y_range, y_kr2, c='g', label='Kernel Regression')
#     plt.plot(y_range, y_svr2, c='r', label='SVR')
    plt.xlabel('bbox_pred')
    plt.ylabel('mAP')
    plt.ylim(0, 0.85)
#     plt.title('Kernel regression versus SVR')
    plt.legend()
    plt.show()
    
    # Visualize learning curves
    plt.figure()
    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(svr, X, y, train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    train_sizes_abs, train_scores_kr, test_scores_kr = \
        learning_curve(kr, X, y, train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    plt.plot(train_sizes, test_scores_svr.mean(1), 'o-', color="r",
             label="SVR")
    plt.plot(train_sizes, test_scores_kr.mean(1), 'o-', color="g",
             label="Kernel Regression")
    plt.yscale("symlog", linthreshy=1e-7)
    plt.ylim(-10, -0.01)
    plt.xlabel("Training size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")
    plt.show()

#   def multivar_regress6(self):
#     X, y = self.regression_data()
#     X = np.array(X)
#     x = X[:,0]
#     y = np.array(y)
#     
#     k0 = smooth.NonParamRegression(x, y, method=npr_methods.SpatialAverage())
#     k0.fit()
#     
#     grid = np.r_[0:2.5:512j]
# #     plt.plot(grid, f(grid), 'r--', label='Reference')
#     plt.plot(x, y, '.', alpha=0.5, label='Data')
# #     plt.legend(loc='best')
#     
#     plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
#     plt.legend(loc='best')
#     
#     yopts = k0(x)
#     res = y - yopts
#     plot_fit.plot_residual_tests(x, yopts, res, 'Spatial Average')
      
  def get_same_Kfrac_stats(self, labels):
    new_stats = defaultdict( dict )
    for data_dict in self._stats.values():
      Kfrac = data_dict['Kfrac']
      if Kfrac in new_stats:
        continue
      d_same_net = [d for d in self._stats.values() if d['Kfrac']==Kfrac]
      for d in d_same_net:
        for label in labels:
          if label in d:
            new_stats[Kfrac][label] = d[label]   
    return new_stats   

  def regression_data_split(self):
    X = []
    y = []
    mAP_label = 'mAP_100_top150'
    X1_label = 'diff_mean_cls_prob'
    X2_label = 'diff_mean_rois'
    stats = self.get_same_Kfrac_stats([mAP_label, X1_label, X2_label])
    for _, d in stats.items():
#       cls_score = d['mismatch_count_cls_score']
#       bbox_pred = d['mismatch_count_bbox_pred']
      cls_prob = d[X1_label]
      diff_mean_rois = d[X2_label]
      mAP = d[mAP_label]
      X.append([cls_prob, diff_mean_rois])
      y.append(mAP)
    return X, y
    
  def regression_data(self):
    X = []
    y = []
    
    for _, d in self._stats.items():
#       cls_score = d['mismatch_count_cls_score']
#       bbox_pred = d['mismatch_count_bbox_pred']
      cls_score = d['diff_mean_cls_prob']
#       bbox_pred = d['diff_mean_bbox_pred']
      mAP = d['mAP_100_top150']
      X.append([cls_score, bbox_pred])
      y.append(mAP)
    return X, y
    
  def multivar_regress2(self):
    X, y = self.regression_data()
    
    #predict is an independent variable for which we'd like to predict the value
    predict= [[0.49, 0.18],[3,3],[1.5,1.0]]
    
    #generate a model of polynomial features
    poly = PolynomialFeatures(degree=2)
    
    #transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
    X_ = poly.fit_transform(X)
    
    #transform the prediction to fit the model type
    predict_ = poly.fit_transform(predict)
    
    #here we can remove polynomial orders we don't want
    #for instance I'm removing the `x` component
#     X_ = np.delete(X_,(1),axis=1)
#     predict_ = np.delete(predict_,(1),axis=1)
    
    #generate the regression object
    clf = linear_model.LinearRegression()
    #preform the actual regression
    clf.fit(X_, y)
    
    print("X_ = ",X_)
    print("predict_ = ",predict_)
    print("Prediction = ",clf.predict(predict_))
    return 2

  def plot(self, plot_data=None, legend_labels=None):
    fig, ax = plt.subplots()
    plt.legend(legend_labels, title=r'fraction of $K_{max}$', loc='upper left')
#     plt.title('Reconstruction Error')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     legend_labels = []
    data_dict = self.build_label_layer_K_dict()
    
    if plot_data:
      n_rows = len(plot_data)
    else:
      n_rows = len(data_dict)
      
    n_columns = 1
    plt_idx = 1
    layers_names = []
     
    for properties in plot_data:
      type_label = properties[0]
      y_label = properties[1]
      percent_fmt = properties[2]
      d = data_dict[type_label]
#     for i, (type_label, d) in enumerate(sorted(data_dict.items())):
#       if plot_type_labels and type_label not in plot_type_labels:
#         continue
      num_layers = len(d)
      num_K = len( list(d.values())[0] )
      plot_data = np.zeros( (num_K, num_layers) )
      ax = plt.subplot(n_rows, n_columns, plt_idx)
      ax.axhline(y=0, color='k', linewidth=0.5, label='_nolegend_')
#       ax.spines['left'].set_position('zero')
#       ax.spines['bottom'].set_position('zero')
      # get rid of the frame
#       for spine in plt.gca().spines.values():
#         spine.set_visible(False)

      plt_idx += 1
 
      layer_idxs = []
      for j, (layer, d2) in enumerate(sorted(d.items())):
#         if j not in [31,33]:
#           continue
        Ks = []
        layer_idxs.append(j)
        layers_names.append(layer.replace('bottleneck_v1/','').replace('block','b').replace('unit_','u'))
        for k, (K, val) in enumerate(sorted(d2.items())):
          plot_data[k,j] = val
          Ks.append(K)
#           plot_data[k,j] = self._stats[type_label][layer][K]

      for k, (K, val) in enumerate(sorted(d2.items())):
        if type_label in ['flops_reduced_count']:
          plt.semilogy(layer_idxs, plot_data[k,:],'.-', linewidth=0.5, markersize=1.0)
        else:
          plt.plot(layer_idxs, plot_data[k,:],'.-', linewidth=0.5, markersize=1.0)
#         legend_labels.append('K=%d'%K)
#         plt.plot(Ks, plot_data[:,j],'o-')
#       legend_labels.append(layer)
#       legend_labels.append(type_label)
#       plt.plot(range(1,num_layers+1), plot_data[0,:],'ro-')
      plt.ylabel(y_label)
      if percent_fmt:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 


#     plt.xlabel('K')
#       plt.ylabel(type_label)
    plt.xticks(layer_idxs, layers_names, rotation='vertical')

#     plt.legend(legend_labels)
    plt.xlabel('layer index')
    plt.show()  

     
  def plot_by_Kfracs(self, plot_type_label=None):
    base_total = 47567455 # total no. parameters in base net
    plot_data = []
    calced_Kfracs = [] 
     
    for net_desc, data_dict in sorted(self._stats.items()):
      for type_label, value in sorted(data_dict.items()):
        if plot_type_label and plot_type_label != type_label:
#         if plot_type_label and plot_type_label not in type_label:
          continue
        if type_label =='var_redux':
          value = int( (base_total - value) / 1000000 )
        Kfrac = net_desc.get_Kfrac() 
        if Kfrac != 0:
          calced_Kfracs.append( Kfrac )
          plot_data.append(value)
        else:
          plt.axhline(y=value, color='r', linewidth=0.5)

        
    calced_Kfracs, plot_data = zip(*sorted(zip(calced_Kfracs, plot_data)))
    plt.ticklabel_format(style='plain')
    plt.plot(calced_Kfracs, plot_data,'-o')
#     plt.plot(Kfracs, plot_data,'o-')
    plt.ylabel('mAP #images=5')
#     plt.ylabel('No. parameters in net $x10^6$')
#     plt.ylabel('mAP')
#     plt.xlabel(r'fraction of $K_{max}$')
    plt.xlabel(r'$K_{frac}$', weight='bold')
    plt.show()  
     
       
  def plot_K_by_layer(self, layers_names, Kfracs, plot_type_label=None):
    plt.title('Profile of layer compressions',fontsize=16)
    legend_labels = []
     
    for ii, (net_desc, data_dict) in enumerate(sorted(self._stats . items())):
      for i, (type_label, value) in enumerate(sorted(data_dict.items())):
        if plot_type_label and type_label not in plot_type_label:
          continue
        num_layers = len(net_desc)
        plot_data = []
        x = range(0,num_layers)
        for layer in layers_names:
          K = net_desc[layer]
          plot_data.append(K)
        plt.plot(x, plot_data,'o-')
        legend_labels.append('%.2f'%Kfracs[ii])
    
    plt.ylabel('K', fontsize=16)
    plt.xlabel('layer', fontsize=16)
    
    layers_names = [name.replace('/bottleneck_v1/conv2','') for name in layers_names]
    plt.xticks(x, layers_names, rotation='vertical')
    plt.legend(legend_labels, title=r'fraction of $K_{max}$')
    plt.show()  
     
  def plot_single_layers(self, layers_names, Kfracs, plot_type_label=None, ylabel=None):
    plt.title('Effects of single layer compressions',fontsize=16)
    legend_labels = []
    num_Kfracs = len(Kfracs)
    xs = range(1,len(layers_names)+1)
    plot_data = np.empty((len(layers_names), num_Kfracs))
     
    for ii, (net_desc, data_dict) in enumerate(sorted(self._stats.items())):
      for i, (type_label, value) in enumerate(sorted(data_dict.items())):
        if plot_type_label and type_label not in plot_type_label:
          continue
        layer_idx = layers_names.index(list(net_desc.keys())[0])
        Kfrac_idx = ii % num_Kfracs
        plot_data[layer_idx, Kfrac_idx] = value        
    
    for Kfrac_idx, Kfrac in enumerate(Kfracs):
      plt.plot(xs, plot_data[:,Kfrac_idx],'o-')
      legend_labels.append('%.2f'%Kfrac)
    legend_labels[0] = 'K=1'  
      
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel('layer', fontsize=16)
    
#     layers_names = [name.replace('/bottleneck_v1/conv2','') for name in layers_names]
#     plt.xticks(x, layers_names, rotation='vertical')
    plt.legend(legend_labels, title=r'fraction of $K_{max}$')
    plt.show()  
     
       
  def plot_correlation(self, type_labels, labels=None):
#     legend_labels = []
    if not labels:
      labels=[]

    for type_label in type_labels:
      xs = []
      ys = []
  
      for net_desc, data_dict in sorted(self._stats . items()):
#         net_desc
        mAP = [data_dict[key] for key in data_dict.keys() if re.match('mAP', key)]
        if type_label in data_dict:
          diff_mean = data_dict[type_label]
          xs.append(mAP[0])
          ys.append(diff_mean)
          print( net_desc )
          labels.append('%.2f'%net_desc.get_Kfrac())
  
      plt.plot(xs, ys,'.')
  #     np.corcoef()
      
      plt.xlabel('mAP', fontsize=16)
      plt.ylabel('mean reconstruction error', fontsize=16)
  #     x1,x2,y1,y2 = plt.axis()
  #     plt.axis((0.7,0.8,y1,y2))
  
      if labels:
        for label, x, y in zip(labels, xs, ys):
          plt.annotate(
              str(label),
              xy=(x, y), xytext=(20, -10),
              textcoords='offset points', ha='right', va='bottom',
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
              )

    plt.legend(['block4 output','block3 output'])
    plt.show()  
     
     
  def data_dict_from_Kfrac(self, Kfrac):
    for net_desc, data_dict in sorted(self._stats.items()):
      if abs(net_desc.get_Kfrac() - Kfrac) < 1.0e-9:
        return data_dict
    return None
        
       
  def get_key_value_match(self, d, match_key_pattern):     
    for key, value in d.items():
      if re.match(match_key_pattern, key):
        return value
    return None
      
      
  def plot_correlation_btw_stats(self, other_stats, type_label='mAP', labels=None):
#     legend_labels = []
    if not labels:
      labels=[]

    xs = []
    ys = []

    for net_desc, data_dict in sorted(self._stats.items()):
      value = self.get_key_value_match(data_dict, type_label)
      if value:
        other_data_dict = other_stats.data_dict_from_Kfrac(net_desc.get_Kfrac())
        if other_data_dict:
          other_value = self.get_key_value_match(other_data_dict, type_label)
          ys.append(value)
          xs.append(other_value)
#           print( net_desc )
          labels.append('%.2f'%net_desc.get_Kfrac())

      plt.plot(xs, ys,'.')
      
  #     x1,x2,y1,y2 = plt.axis()
  #     plt.axis((0.7,0.8,y1,y2))
  
      if labels:
        for label, x, y in zip(labels, xs, ys):
          if label == '0.40':
            xoffset = -20
          else:
            xoffset = 30
            
          if label == '0.60':
            yoffset = 10
          elif label == '0.50':
            yoffset = -20
          else:
            yoffset = -10
            
          plt.annotate(
              str(label),
              xy=(x, y), xytext=(xoffset, yoffset),
              textcoords='offset points', ha='right', va='bottom',
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
              arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
              )

    plt.ylabel(type_label+' #images 200', fontsize=16)
    plt.xlabel(type_label+' #images 4952', fontsize=16)
    nxs = np.vstack((xs,ys))
    corr_coeff = np.corrcoef(nxs)
    plt.text(0.2, 0.65, 'corr coeff = %.3f'%corr_coeff[0,1], fontsize=16, horizontalalignment='center', verticalalignment='center')
    plt.show()  
     
       
  def plot_correlation_btw_mAP_num_imgs(self, min_mAP=0):
    fig, ax = plt.subplots()
    n_rows = 3
    n_columns = 3
    
    def get_mAPs():
      values = [v for v in list(self._stats.values())[0] if 'mAP' in v and 'delta' not in v]
      return sorted(values, key=lambda value: int(value.split('_')[1]))
      
    mAP_labels = get_mAPs()
    top_mAP_label = mAP_labels[-1]
    
    for plt_idx, mAP_label in enumerate(mAP_labels[:-1]):
      xs = []
      ys = []
      for _, data_dict in sorted(self._stats.items()):
        mAP = data_dict[mAP_label]
        top_mAP = data_dict[top_mAP_label]
        if mAP < min_mAP or top_mAP < min_mAP:
          continue
        ys.append(mAP)
        xs.append(top_mAP)
      ax = plt.subplot(n_rows, n_columns, plt_idx+1)
      ax.plot(xs, ys,'.r', markersize=5.0)
      if plt_idx > 5:
        ax.set_xlabel('mAP #images=4952', fontsize=10)
      ax.set_ylabel('mAP #images=%s'%(mAP_label.split('_')[1]), fontsize=10)
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      nxs = np.vstack((xs,ys))
      corr_coeff = np.corrcoef(nxs)#
      plt.text(0.2, 0.65, 'r = %.3f'%corr_coeff[0,1], fontsize=10, weight='bold', horizontalalignment='center', verticalalignment='center')
#       plt.text(0.73, 0.82, 'r = %.3f'%corr_coeff[0,1], fontsize=10, weight='bold', horizontalalignment='center', verticalalignment='center')
      #fit function
      a, b = np.polyfit(np.array(xs), np.array(ys), deg=1)
      f = lambda x: a*x + b
      x = np.array([0,0.85])
      ax.plot(x,f(x),lw=0.2, c="k")
      ax.set_xlim([min_mAP,0.85])
      ax.set_ylim([min_mAP,0.85])
      
    fig.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.06, hspace=0.24, wspace=0.38)
    plt.show()  
     
       
