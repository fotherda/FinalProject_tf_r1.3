'''
Created on 13 Jul 2017

@author: david
'''
import numpy as np
import pickle as pi
import matplotlib.pyplot as plt
import re

from collections import OrderedDict,defaultdict
from matplotlib.ticker import MaxNLocator, FuncFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from scipy import optimize
from scipy.optimize import least_squares, minimize
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods, plot_fit

from functools import total_ordering
# from davelib.utils import get_all_compressible_layers



@total_ordering
class CompressedNetDescription(dict):
   
  def __init__(self, compressed_layers, Ks):
    items = []
    self._Ks = []
    for layer, K in zip(compressed_layers,Ks):
      self[layer] = K
      self._Ks.append(K)
      items.extend((layer, K))
    self._key = tuple(items)
      
#   def __key(self):
#     return self['conv1']
  
  def __key(self):
    return self._key
   
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__key() == other.__key()
    return False
  
  def __hash__(self):
    return hash(self.__key())
  
  def __lt__(self, other):
    return list(self.keys())[0] < list(other.keys())[0]
  
#   def __str__(self):
#     return '\n'.join(map(str, sorted(self.items())))

  def get_Kfrac(self):
    for layer, K in self.items():
      if 'block4' in layer:
        Kfrac = K / 768.0
        break
      elif 'block3' in layer:
        Kfrac = K / 384.0
        break
    return Kfrac

class CompressionStats(object):

  def __init__(self, filename_suffix='', load_from_file=True, all_Kmaxs_dict=None):
    self._filename_suffix = filename_suffix
    self._all_Kmaxs_dict = all_Kmaxs_dict
    if load_from_file: #load from pickle file
      self.load_from_file('CompressionStats_'+filename_suffix+'.pi')
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
    self._stats = pi.load( open( filename, "rb" ) ) 
#     print self._stats   

  def set(self, net_desc, type_label, value):
    self._stats[net_desc][type_label] = value

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
    
    func = getattr(profile_stats, 'total_bytes_'+count_or_frac)
    self.set(net_desc, 'total_bytes_'+count_or_frac, func(base_profile_stats))
    
  def save(self, suffix=None):
    if not suffix:
      suffix = self._filename_suffix
    pi.dump( self._stats, open( 'CompressionStats_%s.pi'%suffix, "wb" ) )

  def add_data_type(self, type_label, values): #values must be in order to match sorted net descriptions
    for i, (K_by_layer_dict, d) in enumerate(sorted(self._stats.items())):
      print(str(K_by_layer_dict['conv1']))
      d[type_label] = values[i]
    
  def merge(self, filename_suffix):
    filename = 'CompressionStats_'+filename_suffix+'.pi'
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
      if len(net_desc) != 1:
        continue
      layer = list(net_desc.keys())[0]
      K = net_desc[layer]
      for type_label, value in d.items():
        new_dict[type_label][layer][K] = value
    return new_dict
  

  
  def multivar_regress(self):
    X, y = self.regression_data()
    X = np.array(X)
    x = X[:,0]
    y = np.array(y)
    
    k0 = smooth.NonParamRegression(x, y, method=npr_methods.SpatialAverage())
    k0.fit()
    
    grid = np.r_[0:2.5:512j]
#     plt.plot(grid, f(grid), 'r--', label='Reference')
    plt.plot(x, y, '.', alpha=0.5, label='Data')
#     plt.legend(loc='best')
    
    plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
    plt.legend(loc='best')
    
    yopts = k0(x)
    res = y - yopts
    plot_fit.plot_residual_tests(x, yopts, res, 'Spatial Average')
    
  def multivar_regress4(self):
    X, y = self.regression_data()
    X = np.array(X)
    x = X[:,0]
    y = np.array(y)
    
#     x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
#     y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

    def piecewise_linear(x, x0, y0, k1, k2):
      return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

    def fun(params, x, y):
      x0 = params[0] 
      y0 = params[1]
      k1 = params[2] 
      k2 = params[3]
      return piecewise_linear(x, x0, y0, k1, k2) - y
    
    
    x0=np.array([1.2,0.74,0.001,-1.0])
#     res = least_squares(fun, x0, args=(x, y), tr_solver='lsmr', method='dogbox', ftol=1e-16, xtol=1e-16, 
#                         verbose=2, bounds=([0,0,-1, -5], [2,2,1,0]))
#     res = least_squares(fun, x0, args=(x, y), verbose=2, tr_solver='lsmr', method='lm')
#     res = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(x, y), verbose=2)
    
#     print(res.x)
    
    p, e = optimize.curve_fit(piecewise_linear, x[:20], y[:20], method='trf', tr_solver='lsmr', verbose=2)
    p, e = optimize.curve_fit(piecewise_linear, x[:10], y[:10])
    p, e = optimize.curve_fit(piecewise_linear, x[:100], y[:100])
    xd = np.linspace(0, 15, 100)
    plt.plot(x, y, ".")
    plt.plot(xd, piecewise_linear(xd, *p))
    plt.show()
  
  def multivar_regress3(self):
#     np.random.seed(0)
# 
#     n_samples = 30
    degrees = [1, 4, 15]
    
#     true_fun = lambda X: np.cos(1.5 * np.pi * X)
#     X = np.sort(np.random.rand(n_samples))
#     y = true_fun(X) + np.random.randn(n_samples) * 0.1
    
    X, y = self.regression_data()
    X = np.array(X)
    X = X[:,0]
    
    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())
    
        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)
    
        # Evaluate the models using crossvalidation
        scores = cross_validation.cross_val_score(pipeline,
            X[:, np.newaxis], y, scoring="mean_squared_error", cv=10)
    
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
#         plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, s=1, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
#         plt.xlim((0, 1))
#         plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()))
    plt.show()

  def regression_data(self):
    X = []
    y = []
    
    for _, d in self._stats.items():
      cls_score = d['diff_mean_cls_score']
      bbox_pred = d['diff_mean_bbox_pred']
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

    plt.legend(legend_labels, title=r'fraction of $K_{max}$')
#     plt.legend(legend_labels)
    plt.xlabel('layer index')
    plt.show()  

     
  def plot_by_Kfracs(self, plot_type_label=None):
    base_total = 47567455 # total no. parameters in base net
    plot_data = []
    calced_Kfracs = [] 
     
    for net_desc, data_dict in sorted(self._stats.items()):
      for type_label, value in sorted(data_dict.items()):
        if plot_type_label and plot_type_label not in type_label:
          continue
        if type_label =='var_redux':
          value = int( (base_total - value) / 1000000 )
        plot_data.append(value)
        calced_Kfracs.append( net_desc.get_Kfrac() )
        
    plt.ticklabel_format(style='plain')
    plt.plot(calced_Kfracs, plot_data,'o-')
#     plt.plot(Kfracs, plot_data,'o-')
    plt.ylabel('No. parameters in net $x10^6$')
#     plt.ylabel('mAP')
    plt.xlabel(r'fraction of $K_{max}$')
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
      
      
  def plot_correlation_btw_stats(self, other_stats, type_label, labels=None):
#     legend_labels = []
    if not labels:
      labels=[]

    xs = []
    ys = []

    for net_desc, data_dict in sorted(self._stats . items()):
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
     
       
