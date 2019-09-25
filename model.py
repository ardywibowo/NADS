from math import exp, log, pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la
from torch.autograd import Variable

from operations import *

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class MixedOp(nn.Module):

  def __init__(self, C, stride, cell):
    super(MixedOp, self).__init__()

    self.op = OPS[cell](C, stride, False)
    if 'pool' in cell:
      self.op = nn.Sequential(self.op, nn.BatchNorm2d(C, affine=False))

  def forward(self, x):
    return self.op(x)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, cells):
    super(Cell, self).__init__()

    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    op_num = 0
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride, cells[op_num])
        self._ops.append(op)
        op_num += 1

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = 0
      for j, h in enumerate(states):
        s_curr = self._ops[offset+j](h)
        s += s_curr
      
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class AffineCoupling(nn.Module):
  def __init__(self, in_channel, cells, filter_size=512, affine=True, multiplier=4, learnable_steps=2):
    super().__init__()

    self.affine = affine

    self.conv_init = nn.Conv2d(in_channel // 2, filter_size, 3, padding=1)
    self.conv_init.weight.data.normal_(0, 0.05)
    self.conv_init.bias.data.zero_()

    self.learnable_cell = Cell(learnable_steps, multiplier, filter_size, filter_size, filter_size//multiplier, cells)
    
    self.conv_zero = ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2)

  def forward(self, input):
    in_a, in_b = input.chunk(2, 1)

    if self.affine:
      net_out = self.conv_init(in_a)
      net_out = self.learnable_cell(net_out, net_out)
      net_out = nn.ReLU(inplace=True)(net_out)
      net_out = self.conv_zero(net_out)

      log_s, t = net_out.chunk(2, 1)
      s = F.sigmoid(log_s + 2)
      out_b = (in_b + t) * s

      logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

    else:
      net_out = self.conv_init(in_a)
      net_out = self.learnable_cell(net_out, net_out)
      net_out = nn.ReLU(inplace=True)(net_out)
      net_out = self.conv_zero(net_out)

      out_b = in_b + net_out
      logdet = None

    return torch.cat([in_a, out_b], 1), logdet

  def reverse(self, output):
    out_a, out_b = output.chunk(2, 1)

    if self.affine:
      net_out = self.conv_init(out_a)
      net_out = self.learnable_cell(net_out, net_out)
      net_out = nn.ReLU(inplace=True)(net_out)
      net_out = self.conv_zero(net_out)

      log_s, t = net_out.chunk(2, 1)
      s = F.sigmoid(log_s + 2)
      in_b = out_b / s - t

    else:
      net_out = self.conv_init(out_a)
      net_out = self.learnable_cell(net_out, net_out)
      net_out = nn.ReLU(inplace=True)(net_out)
      net_out = self.conv_zero(net_out)

      in_b = out_b - net_out

    return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
  def __init__(self, in_channel, cells, affine=True, conv_lu=True, learnable_steps=2):
    super().__init__()

    self.actnorm = ActNorm(in_channel)

    if conv_lu:
      self.invconv = InvConv2dLU(in_channel)

    else:
      self.invconv = InvConv2d(in_channel)

    self.coupling = AffineCoupling(in_channel, cells, affine=affine, learnable_steps=learnable_steps)

  def forward(self, input):
    out, logdet = self.actnorm(input)
    out, det1 = self.invconv(out)
    out, det2 = self.coupling(out)

    logdet = logdet + det1
    if det2 is not None:
      logdet = logdet + det2

    return out, logdet

  def reverse(self, output):
    input = self.coupling.reverse(output)
    input = self.invconv.reverse(input)
    input = self.actnorm.reverse(input)

    return input

def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
  def __init__(self, in_channel, n_flow, cells, learnable_steps=2, split=True, affine=True, conv_lu=True):
    super().__init__()

    squeeze_dim = in_channel * 4

    self.flows = nn.ModuleList()
    for i in range(n_flow):
      self.flows.append(Flow(squeeze_dim, cells[i], affine=affine, conv_lu=conv_lu, learnable_steps=learnable_steps))

    self.split = split

    if split:
      self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

    else:
      self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

  def forward(self, input):
    b_size, n_channel, height, width = input.shape
    squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
    out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

    logdet = 0

    for flow in self.flows:
      out, det = flow(out)
      logdet = logdet + det

    if self.split:
      out, z_new = out.chunk(2, 1)
      mean, log_sd = self.prior(out).chunk(2, 1)
      log_p = gaussian_log_p(z_new, mean, log_sd)
      log_p = log_p.view(b_size, -1).sum(1)

    else:
      zero = torch.zeros_like(out)
      mean, log_sd = self.prior(zero).chunk(2, 1)
      log_p = gaussian_log_p(out, mean, log_sd)
      log_p = log_p.view(b_size, -1).sum(1)
      z_new = out

    return out, logdet, log_p, z_new

  def reverse(self, output, eps=None, reconstruct=False):
    input = output

    if reconstruct:
      if self.split:
        input = torch.cat([output, eps], 1)

      else:
        input = eps

    else:
      if self.split:
        mean, log_sd = self.prior(input).chunk(2, 1)
        z = gaussian_sample(eps, mean, log_sd)
        input = torch.cat([output, z], 1)

      else:
        zero = torch.zeros_like(input)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        z = gaussian_sample(eps, mean, log_sd)
        input = z

    for flow in self.flows[::-1]:
      input = flow.reverse(input)

    b_size, n_channel, height, width = input.shape

    unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
      b_size, n_channel // 4, height * 2, width * 2
    )

    return unsqueezed

class Network(nn.Module):

  def __init__(self, in_channel, n_flow, n_block, n_bins, cells, affine=True, conv_lu=True, learnable_steps=2, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._in_channel = in_channel
    self._n_flow = n_flow
    self._n_block = n_block
    self._n_bins = n_bins
    self._learnable_steps = learnable_steps
    self._multiplier = multiplier
    self._cells = cells

    self.blocks = nn.ModuleList()
    n_channel = in_channel
    for i in range(n_block - 1):
      self.blocks.append(Block(n_channel, n_flow, cells[i], affine=affine, conv_lu=conv_lu, learnable_steps=learnable_steps))
      n_channel *= 2
    self.blocks.append(Block(n_channel, n_flow, cells[-1], split=False, affine=affine, learnable_steps=learnable_steps))

  def forward(self, input):
    log_p_sum = 0
    logdet = 0
    out = input
    z_outs = []

    for block in self.blocks:
      out, det, log_p, z_new = block(out)
      z_outs.append(z_new)

      logdet = logdet + det

      if log_p is not None:
        log_p_sum = log_p_sum + log_p

    return log_p_sum, logdet, z_outs

  def reverse(self, z_list, reconstruct=False):

    for i, block in enumerate(self.blocks[::-1]):
      if i == 0:
        input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
      else:
        input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

    return input
