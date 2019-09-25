import argparse
import glob
import logging
import os
import pickle
import random
import sys
import time
from math import log, pi, sqrt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
import torchvision
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils as tvutils
from tqdm import tqdm

from utils import calc_z_shapes, likelihood_loss
from model import Network as EnsembleNetwork

import multiprocessing
multiprocessing.set_start_method("spawn", True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='NADS Evaluation')
parser.add_argument('--n_flow', default=32, type=int, help='number of flows in each block')
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', action='store_true', default=True, help='use affine coupling instead of additive')
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')

parser.add_argument('--weights_name', default="mnist_1", type=str, help='weight name')
parser.add_argument('--gpu', default="0", type=int, help='gpu')


def sample_data(path, image_size):
  transform = transforms.Compose(
    [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
    ]
  )

  dataset = datasets.ImageFolder(path, transform=transform)
  loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

  return loader

def sample_data_one_channel(path, image_size):
  transform = transforms.Compose(
    [
      transforms.Grayscale(),
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
    ]
  )

  dataset = datasets.ImageFolder(path, transform=transform)
  loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

  return loader

def sample_mnist(image_size):
  transform = transforms.Compose(
    [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
    ]
  )

  dataset = torchvision.datasets.MNIST("../../data/", train=False, transform=transform, target_transform=None, download=True)
  loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

  return loader

def sample_kmnist(image_size):
  transform = transforms.Compose(
    [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
    ]
  )

  dataset = torchvision.datasets.KMNIST("../../data/", train=False, transform=transform, target_transform=None, download=True)
  loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

  return loader

def sample_fmnist(image_size):
  transform = transforms.Compose(
    [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
    ]
  )

  dataset = torchvision.datasets.FashionMNIST("../../data/", train=False, transform=transform, target_transform=None, download=True)
  loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

  return loader

def compute_likelihoods(args, model, dset, weights_name):
  if dset == "mnist":
    dataset = sample_mnist(args.img_size)
  elif dset == "kmnist":
    dataset = sample_kmnist(args.img_size)
  elif dset == "fmnist":
    dataset = sample_fmnist(args.img_size)
  elif dset == "notmnist":
    dataset = sample_data_one_channel("notMNIST_large/", args.img_size)

  n_bins = 2. ** args.n_bits

  log_likelihoods = []
  for i, datapoint in enumerate(dataset):
    image, _ = datapoint
    image = image.to(device)

    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
    logdet = logdet.mean()
    loss, log_p, log_det = likelihood_loss(log_p, logdet, args.img_size, n_bins)
    log_likelihoods.append(log_p.item() + log_det.item())

    print('#{} Loss: {}; logP: {}; logdet: {}'.format(i, loss.item(), log_p.item(), log_det.item()))
    
    if i % 20 == 0:
      np.save(dset + "_" + weights_name + "_likelihoods", np.array(log_likelihoods))

  np.save(dset + "_" + weights_name + "_likelihoods", np.array(log_likelihoods))

if __name__ == '__main__':
  args = parser.parse_args()
  torch.cuda.set_device(args.gpu)
  print(args)

  # Define model and loss criteria
  n_bins = 2. ** args.n_bits
  weights_name = args.weights_name
  
  with open(weights_name + '.pkl', 'rb') as fp:
    genotype = pickle.load(fp)

  model_single = EnsembleNetwork(1, args.n_flow, args.n_block, n_bins, genotype, affine=True, conv_lu=not args.no_lu, learnable_steps=2)
  model = model_single
  model = model.to(device)

  model.load_state_dict(torch.load(weights_name + ".pt", map_location="cuda:{}".format(args.gpu)))

  dset = "mnist"
  compute_likelihoods(args, model, dset, weights_name)

  dset = "kmnist"
  compute_likelihoods(args, model, dset, weights_name)

  dset = "fmnist"
  compute_likelihoods(args, model, dset, weights_name)
  
  dset = "notmnist"
  compute_likelihoods(args, model, dset, weights_name)
