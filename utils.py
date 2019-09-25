import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from math import log, pi, sqrt

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  os.mkdir(os.path.join(path, 'samples'))
  os.mkdir(os.path.join(path, 'checkpoint'))
  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
  z_shapes = []

  for i in range(n_block - 1):
    input_size //= 2
    n_channel *= 2

    z_shapes.append((n_channel, input_size, input_size))

  input_size //= 2
  z_shapes.append((n_channel * 4, input_size, input_size))

  return z_shapes

def likelihood_loss(log_p, logdet, image_size, n_bins):
  n_pixel = image_size * image_size * 3

  loss = -log(n_bins) * n_pixel
  loss = loss + logdet + log_p

  return (
    (-loss / (log(2) * n_pixel)).cuda().mean(),
    (log_p / (log(2) * n_pixel)).cuda().mean(),
    (logdet / (log(2) * n_pixel)).cuda().mean(),
  )

def likelihood_loss_variance(log_p, logdet, image_size, n_bins, approx_samples):
  # log_p = calc_log_p([z_list])
  n_pixel = image_size * image_size * 3

  log_p = log_p.view(approx_samples, -1)
  logdet = logdet.view(approx_samples, -1)

  loss = -log(n_bins) * n_pixel
  loss = loss + logdet + log_p

  variance = (-loss / (log(2) * n_pixel)).cuda().std(axis=0) ** 2

  return variance.mean()
