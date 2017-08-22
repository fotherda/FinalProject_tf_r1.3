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
    
  def mAP(self, num_imgs_list, rpn_post_nms_top_n, sess): #get mean average precision
    ret_dict = {}
    if len(num_imgs_list) == 1:
      num_imgs = num_imgs_list[0]
      if num_imgs == 0:
        ret_dict[num_imgs] = 0
        return ret_dict
    
    if os.path.isfile(mAP_filename):
      with open(mAP_filename, 'rb') as f:
        mAP_dict = pi.load(f)
        if rpn_post_nms_top_n in mAP_dict:
          for num_imgs in num_imgs_list:
            if num_imgs in mAP_dict[rpn_post_nms_top_n]:
              ret_dict[num_imgs] = mAP_dict[rpn_post_nms_top_n][num_imgs]
      return ret_dict
    else:
      print('Can\'t find: ' + mAP_filename)
      mAP_dict = defaultdict( dict )

    img_mAP_dict = run_test_metric(num_imgs_list, self._base_net, sess)
    for num_imgs, mAP in img_mAP_dict.items():
      mAP_dict[rpn_post_nms_top_n][num_imgs] = mAP
    
    pi.dump( mAP_dict, open( mAP_filename, "wb" ) )
    
#     max_num_imgs = sorted(num_imgs_list)[-1]
#     return img_mAP_dict[max_num_imgs]
    return img_mAP_dict
    