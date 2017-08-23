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

from contextlib import contextmanager
from davelib.voc_img_sampler import VOCImgSampler
from model.test import test_net, test_net_with_sample
from datasets.factory import get_imdb

  
def run_test_metric(num_imgs_list, net, sess, filename=None):
  imdb = get_imdb('voc_2007_test')
  if not filename:
    filename ='default/res101_faster_rcnn_iter_110000'
  
  f = io.BytesIO()
#   with stdout_redirector(f): #this stops some meaningless info on stdout
  if len(num_imgs_list)==1:
    num_imgs = num_imgs_list[0]
    if num_imgs == len(imdb.image_index):
      mAP = test_net(sess, net, imdb, filename, max_per_image=100)
      mAP_dict = {num_imgs: mAP}
    else:
      sampler = VOCImgSampler()
      sample_images = sampler.get_imgs(num_imgs)
      mAP_dict = test_net_with_sample(sess, net, imdb, filename, sample_images, 
                                 max_per_image=100)
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
        super(UnbufferedTextIOWrapper, self).write(s)
        self.flush()

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
  for idx, var in enumerate(tf.global_variables()):
#   for idx, var in enumerate(tf.get_default_graph().as_graph_def().node):
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

