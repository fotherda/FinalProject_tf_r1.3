'''
Created on 20 Sep 2017

@author: david
'''
import pickle as pi

from collections import defaultdict
from davelib.utils import TimingResults

def plot_results(filename):
  
  timing_results = TimingResults(filename)          

  for label, times in timing_results._times_dict.items():
    for t in times:
      print('%s: %.2f'%(label, t))
      