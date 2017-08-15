'''
Created on 8 Aug 2017

@author: david
'''
import pickle as pi
import os
from davelib.utils import run_test_metric
from collections import defaultdict

mAP_filename = 'base_net_mAPs.pi'

class BaseNetWrapper(object):

  def __init__(self, base_resnet):
    self._base_net = base_resnet
    
  def mAP(self, num_imgs, rpn_post_nms_top_n, sess): #get mean average precision
    if num_imgs == 0:
      return 0
    if os.path.isfile(mAP_filename):
      with open(mAP_filename, 'rb') as f:
        mAP_dict = pi.load(f)
        if rpn_post_nms_top_n in mAP_dict:
          if num_imgs in mAP_dict[rpn_post_nms_top_n]:
            return mAP_dict[rpn_post_nms_top_n][num_imgs]
    else:
      print('Can\'t find: ' + mAP_filename)
      mAP_dict = defaultdict( dict )

    mAP = run_test_metric(num_imgs, self._base_net, sess)
    mAP_dict[rpn_post_nms_top_n][num_imgs] = mAP
    
    pi.dump( mAP_dict, open( mAP_filename, "wb" ) )
    
    return mAP
    