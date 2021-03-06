'''
Created on 29 Jul 2017

@author: david
'''
import sys
import numpy as np
import tensorflow as tf
import ctypes
import io
import os
import tempfile
import datetime
import pickle as pi

from contextlib import contextmanager
from davelib.voc_img_sampler import VOCImgSampler
from model.test import test_net, test_net_with_sample
from datasets.factory import get_imdb
from utils.timer import Timer
from collections import defaultdict


class colour:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
        cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

def run_test_metric(num_imgs_list, net, sess, filename=None):
  imdb = get_imdb('voc_2007_test')
  if not filename:
    filename ='default/res101_faster_rcnn_iter_110000'
  
  with timer('run_test_metric - mAP calc'):
    f = io.BytesIO()
    with stdout_redirector(f): #this stops some meaningless info on stdout
      if len(num_imgs_list)==1:
        num_imgs = num_imgs_list[0]
        if num_imgs == len(imdb.image_index):
          mAP = test_net(sess, net, imdb, filename, max_per_image=100)
          mAP_dict = {num_imgs: mAP}
        else:
          sampler = VOCImgSampler()
          sample_images = sampler.get_imgs(num_imgs)
          sample_names_dict = { num_imgs: sample_images }
          mAP_dict = test_net_with_sample(sess, net, imdb, filename, sample_images, 
                                     max_per_image=100, sample_names_dict=sample_names_dict)
      else:
        sampler = VOCImgSampler()
        sample_names_dict = sampler.get_nested_img_lists(num_imgs_list)
        largest_num_imgs = (sorted(num_imgs_list))[-1]
        sample_images = sample_names_dict[largest_num_imgs]
        mAP_dict = test_net_with_sample(sess, net, imdb, filename, sample_images, 
                                   max_per_image=100, sample_names_dict=sample_names_dict)
      return mAP_dict



libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

class TimingResults():
  def __init__(self, filename=None):
    if filename:
      self.load_from_file(filename)
    else:
      self._times_dict = defaultdict( list )
  
  def load_from_file(self, filename):  
    self._times_dict = pi.load( open( filename, "rb") ) 

  def save(self, filename):  
    pi.dump( self._times_dict, open( filename, "wb") ) 
  
  def add_time(self, label, time):
    self._times_dict[label].append( time )
    
class timer:
  def __init__(self, desc='', timing_results=None):
    self._t = Timer() 
    self._desc = desc  
    self._timing_results = timing_results
  def __enter__(self):
    self._t.tic()
    return self
  def __exit__(self, type, value, traceback):
    self._t.toc()
    print(self._desc + ' took: {:.3f}s' .format( self._t.diff))
    if self._timing_results:
      self._timing_results.add_time(self._desc, self._t.diff)

  def elapsed(self):
    return self._t.diff
    
@contextmanager
def timer_func(desc=''):
  _t = Timer()
  _t.tic()
  yield
  _t.toc()
  print(desc + ' took: {:.3f}s' .format( _t.diff))
    

@contextmanager
def stdout_redirector(stream):
    class UnbufferedTextIOWrapper(io.TextIOWrapper):
      def write(self, s):
        super(UnbufferedTextIOWrapper, self).write(s)
        self.flush()

    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = UnbufferedTextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
    
    #if you want to see what came down stderr run this:
    #print(f.getvalue().decode('utf-8'))


@contextmanager
def stderr_redirector(stream):
    class UnbufferedTextIOWrapper(io.TextIOWrapper):
      def write(self, s):
        self.flush()
        super(UnbufferedTextIOWrapper, self).write(s)

    # The original fd stderr points to. Usually 1 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stderr = UnbufferedTextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    # Save a copy of the original stderr fd in saved_stderr_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stderr to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        # Yield to caller, then redirect stderr back to the saved fd
        yield
        _redirect_stderr(saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)


class RedirectStdStreams(object):
  def __init__(self, stdout=None, stderr=None):
      self._stdout = stdout or sys.stdout
      self._stderr = stderr or sys.stderr

  def __enter__(self):
      self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
      self.old_stdout.flush(); self.old_stderr.flush()
      sys.stdout, sys.stderr = self._stdout, self._stderr

  def __exit__(self, exc_type, exc_value, traceback):
      self._stdout.flush(); self._stderr.flush()
      sys.stdout = self.old_stdout
      sys.stderr = self.old_stderr
      
def test_stderr_redirector():
  libc = ctypes.CDLL(None)
  f = io.BytesIO()

  with stderr_redirector(f):
    sys.stderr.write("fatal error\n")
    print('foobar', file=sys.stderr)
    print(12, file=sys.stderr)
    libc.puts(b'this comes from C')
    os.system('echo and this is from echo')
    
  print('Got stdout: "{0}"'.format(f.getvalue().decode('utf-8')), file=sys.stderr)


def show_all_variables(show, *args):
  total_count = 0
#   for idx, var in enumerate(tf.get_default_graph().as_graph_def().node):
  for idx, var in enumerate(tf.global_variables()):
#   for idx, op in enumerate(tf.get_default_graph().get_operations()):
    shape = (0)
    if args:
      for name_filter in args:
        if name_filter not in var.name:
          continue
        else:
          shape = var.get_shape()
    else:
      shape = var.get_shape()
#       if 'shape' in var.attr.keys():
#         for a in var.attr['shape'].shape.dim:
#           if int(a.size) < 0:
#             print(var.name)
            
#         print(var.name, [int(a.size) for a in var.attr['shape'].shape.dim])
#       shape = var.attr['shape'].shape.dim[1].size
    
#     for s in shape:
#       if s < 1:
#         print(var.name)
    count = np.prod(shape)
      
    if show and count>0:
      print("[%2d] %s %s = %s" % (idx, var.name, shape, count))
    total_count += int(count)
  if show:
    print("[Total] variable size: %s" % "{:,}".format(total_count))
  return total_count

def show_all_nodes(show, *args):
  total_count = 0
  for idx, var in enumerate(tf.get_default_graph().as_graph_def().node):
    shape = (0)
#     if args:
#       for name_filter in args:
#         if name_filter not in var.name:
#           continue
#         else:
#           shape = var.get_shape()
#     else:
#       shape = var.get_shape()
#       if 'shape' in var.attr.keys():
#         for a in var.attr['shape'].shape.dim:
#           if int(a.size) < 0:
#             print(var.name)
            
#         print(var.name, [int(a.size) for a in var.attr['shape'].shape.dim])
#       shape = var.attr['shape'].shape.dim[1].size
    
#     for s in shape:
#       if s < 1:
#         print(var.name)
#     count = np.prod(shape)
    
    if show:
#     if show and count>0:
      print("[%2d] %s" % (idx, var.name))
#     total_count += int(count)
#   if show:
#     print("[Total] variable size: %s" % "{:,}".format(total_count))
  return total_count

def remove_net_suffix(input_str, net_root):
  # nasty function to convert e.g. resnet_v1_101_2/block2/unit_1/bottleneck_v1/
  # to                             resnet_v1_101/block2/unit_1/bottleneck_v1/
  # hack to deal with the fact tf adds suffix to scope original name
  idx = input_str.find(net_root)
  if idx == -1:
    return input_str
  else:
    idx = input_str.index('/')
    return net_root + input_str[idx:]
