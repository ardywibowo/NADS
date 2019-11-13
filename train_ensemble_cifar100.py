import argparse
import glob
import logging
import os
import sys
import time
from math import log, pi, sqrt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms, utils
from torchvision import utils as tvutils
from tqdm import tqdm

import utils
from utils import calc_z_shapes, likelihood_loss
from model_search import Network as SearchNetwork
from model import Network as EnsembleNetwork


import random
import pickle

import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='NADS Ensemble')

# Training Settings
parser.add_argument('--learning_rate', type=float, default=1e-5, help='init learning rate')
parser.add_argument('--batch', default=4, type=int, help='batch size')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP-CIFAR', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

# Architecture settings
parser.add_argument('--n_flow', default=32, type=int, help='number of flows in each block')
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', action='store_true', default="True", help='use affine coupling instead of additive')

# Data settings
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--img_size', default=64, type=int, help='image size')

# Sample settings
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
args = parser.parse_args()

# Define Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

def sample_cifar100(batch_size, image_size):
  transform = transforms.Compose(
    [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor()
    ]
  )

  dataset = torchvision.datasets.CIFAR100("../../data/", transform=transform, target_transform=None, download=True)
  loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)

  loader = iter(loader)

  while True:
    try:
      yield next(loader)
    except StopIteration:
      print("Finished Dataset")
      loader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=4
      )
      loader = iter(loader)
      yield next(loader)

def main():
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  print(args)

  seed = random.randint(1, 100000000)
  print(seed)

  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled = True

  n_channels = 3
  n_bins = 2. ** args.n_bits

  # Define model and loss criteria
  model = SearchNetwork(n_channels, args.n_flow, args.n_block, n_bins, affine=args.affine, conv_lu=not args.no_lu)
  model = nn.DataParallel(model, [args.gpu])
  model.load_state_dict(torch.load("architecture.pt", map_location="cuda:{}".format(args.gpu)))
  model = model.module
  genotype = model.sample_architecture()

  with open(args.save + '/genotype.pkl', 'wb') as fp:
    pickle.dump(genotype, fp)

  model_single = EnsembleNetwork(n_channels, args.n_flow, args.n_block, n_bins, genotype, affine=args.affine, conv_lu=not args.no_lu)
  model = model_single
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

  dataset = iter(sample_cifar100(args.batch, args.img_size))

  # Sample generated images
  z_sample = []
  z_shapes = calc_z_shapes(n_channels, args.img_size, args.n_flow, args.n_block)
  for z in z_shapes:
    z_new = torch.randn(args.n_sample, *z) * args.temp
    z_sample.append(z_new.to(device))

  with tqdm(range(args.iter)) as pbar:
    for i in pbar:
      # Training procedure
      model.train()

      # Get a random minibatch from the search queue with replacement
      input, _ = next(dataset)
      input = Variable(input, requires_grad=False).cuda(non_blocking=True)

      log_p, logdet, _ = model(input + torch.rand_like(input) / n_bins)

      logdet = logdet.mean()
      loss, _, _ = likelihood_loss(log_p, logdet, args.img_size, n_bins)

      # Optimize model
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      pbar.set_description(
        "Loss: {}".format(loss.item())
      )

      # Save generated samples
      if i % 100 == 0:
        with torch.no_grad():
          tvutils.save_image(
            model_single.reverse(z_sample).cpu().data,
            "{}/samples/{}.png".format(args.save, str(i + 1).zfill(6)),
            normalize=False,
            nrow=10,
          )

      # Save checkpoint
      if i % 1000 == 0:
        utils.save(model, os.path.join(args.save, 'latest_weights.pt'))

if __name__ == '__main__':
  main() 
