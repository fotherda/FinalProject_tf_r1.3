'''
Created on 18 Aug 2017

@author: david
'''
import sys
import pickle as pi
import matplotlib.pyplot as plt

from collections import OrderedDict,defaultdict
from davelib.layer_name import compress_label
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter



class OptimisationResults():
  def __init__(self, expected_efficiency_delta, actual_perf_delta, expected_perf_delta,
               net_desc, net_change, perf_label, efficiency_label):
    self._expected_efficiency_delta = expected_efficiency_delta
    self._actual_perf_delta = actual_perf_delta
    self._expected_perf_delta = expected_perf_delta
    self._net_desc = net_desc
    self._net_change = net_change
    self._perf_label = perf_label
    self._efficiency_label = efficiency_label
  
  
def plot_results_from_file(filename):
  results = pi.load( open( filename, "rb" ) )
  plot_results(results)

def plot_results(opt_results_list):
  cum_efficiency = 0
  cum_act_perf = 0
  cum_exp_perf = 0
  
  plot_effic = []
  plot_act_perf = []
  plot_exp_perf = []
  xs =[]
  data_labels = []
  
  def abrev_label(res):
    layer = res._net_change._layer
    label = layer.replace('bottleneck_v1/','').replace('unit_','u').replace('/conv2','').\
                  replace('block','block').replace('/',' / ')  
    return label + '   K:%d\u2192%d'%(res._net_change._K_old, res._net_change._K_new)
  
  for i, res in enumerate(opt_results_list):
    cum_efficiency += res._expected_efficiency_delta
    cum_act_perf += res._actual_perf_delta
    cum_exp_perf += res._expected_perf_delta
#     cum_act_perf += res._actual_perf_delta
#     cum_exp_perf += res._expected_perf_delta
    
    plot_effic.append(cum_efficiency)
    plot_act_perf.append(cum_act_perf)
    plot_exp_perf.append(cum_exp_perf)
    xs.append(i+1)

    data_labels.append( abrev_label(res) )

    
  fig, ax = plt.subplots(figsize=(12,7))
  
  ax = plt.subplot(3, 1, 1)
  ys = plot_effic
  ax.plot(xs, ys,'o-')
  plt.ylabel(res._efficiency_label.replace('_',' '), fontsize=12)
  ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:.1%}'.format(x)))
  ax.text(.4,.8,'Efficiency Metric', ha='center', transform=ax.transAxes, color='r', fontsize=16)
  ax.set_xlim(xmin=0, xmax=len(xs)+1)
  ax.xaxis.set_major_locator(plt.NullLocator())
  
  ax = plt.subplot(3, 1, 2)
  ax.plot(xs, plot_act_perf,'.-', label='actual')
  ax.plot(xs, plot_exp_perf,'.-', label='expected')
  plt.ylabel(res._perf_label.replace('_',' '), fontsize=12)
  plt.xlabel('model compression iteration \u27f6', fontsize=14)
  plt.xticks(xs, data_labels, rotation='vertical')
  ax.text(.4,.8,'Performance Metric', ha='center', transform=ax.transAxes, color='r', fontsize=16)
  
#   ax = plt.subplot(3, 1, 3)
#   ax.plot(xs, plot_exp_perf,'o-')
#   plt.ylabel(res._perf_label.replace('_',' '), fontsize=12)
#   plt.xlabel('model compression iteration \u27f6', fontsize=14)
#   plt.xticks(xs, data_labels, rotation='vertical')
#   ax.text(.4,.8,'Performance Metric', ha='center', transform=ax.transAxes, color='r', fontsize=16)
  
  ax.set_xlim(xmin=0, xmax=len(xs)+1)
  ax.set_ylim(ymin=0, ymax=2.0)
  plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.35, hspace=0.07)
  plt.show()  

     
class OptimiseCompression(object):
  
  def __init__(self, compression_stats):
    self._compression_stats = compression_stats #must be single layer compressions

  
      
      
      
      
      
      