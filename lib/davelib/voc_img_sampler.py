'''
Created on 16 Jul 2017

@author: david
'''
import pickle as pi
import numpy as np
import collections
import copy
import sys, os

cachefile = 'data/VOCdevkit2007/annotations_cache/annots.pkl'
sample_file = 'sample_imgs.pi'

class VOCImgSampler:
  """ a completely non-optimized class to do balanced subsampling of the test images where
      we try to ensure the relative numbers of images of each class are the same in the 
      subsample as they are in the full data set.
  """
  def __init__(self):
    with open(cachefile, 'rb') as f:
      try:
        self._recs = pi.load(f)
      except:
        self._recs = pi.load(f, encoding='bytes')

    self._class_img_dict = collections.defaultdict(list)
    self._img_class_dict = collections.defaultdict(set)
    
    for imagename, annos in self._recs.items():
      for anno in annos:
        classname = anno['name']
        difficult = anno['difficult']
        if not difficult: #the evaluation ignores difficult cases so no point in using those images
          self._class_img_dict[classname].append(imagename) 
          self._img_class_dict[imagename].add(classname) 
    
    self._target_ratios = self.get_ratios(self._class_img_dict)  
        
  def get_ratios(self, class_img_dict):      
    sum_num_imgs = 0
    for classname, img_list in class_img_dict.items():
#       print(classname + ': ' + str(len(img_list)))
      sum_num_imgs += len(img_list)

    ratios = {}
    for classname, img_list in class_img_dict.items():
      ratios[classname] = float(len(img_list)) / sum_num_imgs
    return ratios
        
  
  def get_samples(self, num_imgs, sample_names):
    if num_imgs == len(self._img_class_dict): #use all imgs
      all_sample_names = list(self._img_class_dict.keys())
      for sample in all_sample_names:
        if sample not in sample_names:
          sample_names.append(sample)
      return sample_names    

    while len(sample_names) < num_imgs:
      max_ = 0
      next_class=None
      for classname, target_ratio in self._target_ratios.items():
        act_ratio = self._act_ratios[classname]
        deficit = target_ratio - act_ratio
        if deficit > max_:
          next_class = classname
          max_ = deficit
          
      poss_imgs = set(self._class_img_dict[next_class]) #get all imgs containing nextclass object
      poss_imgs = list(poss_imgs) #remove duplicates
      
      min_deficit_img = None
      min_deficit = sys.float_info.max
      for poss_img in poss_imgs:
        if poss_img in sample_names:
          continue
        # calc the diff/deficit between target and act if we add this img
        copy_dict = copy.deepcopy(self._sample_class_img_dict)
        for c in self._img_class_dict[poss_img]:
          copy_dict[c].append(poss_img)
      
        self._act_ratios = self.get_ratios(copy_dict)  
        sum_deficit = self.get_sum_deficit()
        if sum_deficit < min_deficit:
          min_deficit = sum_deficit
          min_deficit_img = poss_img
        
      if min_deficit_img:
        sample_names.append(min_deficit_img)
        for c in self._img_class_dict[min_deficit_img]:
          self._sample_class_img_dict[c].append(min_deficit_img)
      self._act_ratios = self.get_ratios(self._sample_class_img_dict)  
    return sample_names

  def get_nested_img_lists(self, num_imgs_list):
    sample_names_dict = {}
    sample_names = []
    for num_imgs in sorted(num_imgs_list):
      sample_names = self.get_imgs(num_imgs, sample_names)
      sample_names_dict[num_imgs] = sample_names
    return sample_names_dict

  def get_imgs(self, num_imgs, sample_names=[]): 
    if os.path.isfile(sample_file):
      with open(sample_file, 'rb') as f:
        sample_names_dict = pi.load(f)
        if num_imgs not in sample_names_dict:
          sample_names_dict[num_imgs] = sample_names
        else:
          return sample_names_dict[num_imgs]
    else:
      print('Can\'t find: ' + sample_file)
      sample_names_dict = {}
      sample_names_dict[num_imgs] = sample_names
          
    self._act_ratios = dict.fromkeys(self._target_ratios.keys(), 0)

    self._sample_class_img_dict = {}
    for key in self._target_ratios.keys():
      self._sample_class_img_dict[key] = []
    sample_names = self.get_samples(num_imgs, sample_names)

    if False: #for comparison just use random sample
      self.use_rnd_sample(num_imgs)
      
    self.print_results()
    pi.dump( sample_names_dict, open( sample_file, "wb" ) )
    return sample_names
  
  def print_results(self):
    sum_deficit = 0      
    for classname, target_ratio in self._target_ratios.items():
      act_ratio = self._act_ratios[classname]
      deficit = target_ratio - act_ratio
      sum_deficit += abs(deficit)
      print(str(deficit)+' = '+str(target_ratio)+' - '+str(act_ratio) + '  ' + classname)
    print('sum_deficit='+str(sum_deficit))    
  
  def use_rnd_sample(self, num_imgs):
    sample_names = np.random.choice(self._recs.keys(), size=num_imgs, replace=False)
    for name in sample_names:
      for c in self._img_class_dict[name]:
        self._sample_class_img_dict[c].append(name)
    self._act_ratios = self.get_ratios(self._sample_class_img_dict)  

  def get_sum_deficit(self):
    sum_deficit = 0      
    for classname, target_ratio in self._target_ratios.items():
      act_ratio = self._act_ratios[classname]
      deficit = target_ratio - act_ratio
      sum_deficit += abs(deficit)
    return sum_deficit
