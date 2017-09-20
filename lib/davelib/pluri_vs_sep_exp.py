'''
Created on 20 Sep 2017

@author: david
'''
import matplotlib.pyplot as plt

from davelib.utils import TimingResults

def plot_results(filename):
  timing_results = TimingResults(filename)          

  for label, times in timing_results._times_dict.items():
    for t in times:
      print('%s: %.2f'%(label, t))
      
  sep_create = timing_results._times_dict['SeparableNet create_architecture']
  sep_assign = timing_results._times_dict['SeparableNet assign_weights']
  sep_run = timing_results._times_dict['SeparableNet run_performance_analysis']

  pluri_create = timing_results._times_dict['PluripotentNet create_architecture']
  pluri_assign = timing_results._times_dict['PluripotentNet assign_weights']
  pluri_run = timing_results._times_dict['PluripotentNet run_performance_analysis']

  fig = plt.figure(figsize=(10,3))
  ax = fig.add_subplot(111)
  colors ='rgc'
  h = 0.8
  
  x = 0
  for (c, a, r) in zip(sep_create, sep_assign, sep_run):
    ax.barh(1, c, height=h, align='center', left=x, color=colors[0], edgecolor='w')
    x += c                     
    ax.barh(1, a, height=h, align='center', left=x, color=colors[1], edgecolor='w')
    x += a                     
    ax.barh(1, r, height=h, align='center', left=x, color=colors[2], edgecolor='w')
    x += r
                         
  x = 0
  ax.barh(0, pluri_create[0], height=h, align='center', left=x, color=colors[0], edgecolor='w')
  x += c                     
  ax.barh(0, pluri_assign[0], height=h, align='center', left=x, color=colors[1], edgecolor='w')
  x += a                     
  for r in pluri_run:
    ax.barh(0, r, height=h, align='center', left=x, color=colors[2], edgecolor='w')
    x += r
    
  y = 0
  x = 2
  xoffset = -8
  yoffset = -0.8
  plt.annotate('create\narchitecture', xy=(x, y), xytext=(x+xoffset, y+yoffset),
    ha='center', va='top', fontsize=12,
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

  x = 4.5
  xoffset = 0
  plt.annotate('assign\nweights', xy=(x, y), xytext=(x+xoffset, y+yoffset),
    ha='center', va='top', fontsize=12,
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

  x = 7
  xoffset = 7
  plt.annotate('run\ninference', xy=(x, y), xytext=(x+xoffset, y+yoffset),
    ha='center', va='top', fontsize=12,
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

                         
  ax.set_yticks([0, 1])
  ax.set_yticklabels(('Pluripotent Net','Standard Net'), fontsize=16)
  ax.set_ylim([-2.1,1.8])
  ax.set_xlabel('Time in secs', fontsize=14)
  plt.subplots_adjust(left=0.18, right=0.99, top=0.98, bottom=0.2)

  plt.show()
      
      