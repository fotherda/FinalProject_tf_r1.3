'''
Created on 18 Aug 2017

@author: david
'''
import sys
import pickle as pi
import matplotlib.pyplot as plt

from collections import OrderedDict,defaultdict


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
  cum_expected_perf_delta = 0
  
  plot_effic = []
  plot_act_perf = []
  xs =[]
  
  for i, res in enumerate(opt_results_list):
    cum_efficiency += res._expected_efficiency_delta
    cum_act_perf += res._actual_perf_delta
    cum_expected_perf_delta += res._expected_perf_delta
    
    plot_effic.append(cum_efficiency)
    plot_act_perf.append(cum_act_perf)
    xs.append(i+1)
    
  fig, ax = plt.subplots(figsize=(20,3))
  
  ax = plt.subplot(2, 1, 1)
  ax.plot(xs, plot_effic,'o-')
  plt.ylabel(res._efficiency_label.replace('_',' '), fontsize=16)
  plt.xlabel('model compression iteration', fontsize=16)

  ax = plt.subplot(2, 1, 2)
  ax.plot(xs, plot_act_perf,'o-')
  plt.ylabel(res._perf_label.replace('_',' '), fontsize=16)
  
#     ax.set_ylim(ymin=ymin)
#     ax.set_xlim(xmin=0, xmax=len(xs)+1)
#         
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
# #     plt.xticks(xs, layer_names, rotation=45)
#     plt.xticks(xs, layer_names, rotation='vertical')
#     plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.45)
  plt.show()  

     
class OptimiseCompression(object):
  
  def __init__(self, compression_stats):
    self._compression_stats = compression_stats #must be single layer compressions

  
      
      
      
      
      
      