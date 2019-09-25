import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    ActNorm(C, logdet=False)
    ),
}

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
  def __init__(self, in_channel, logdet=True):
    super().__init__()

    self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
    self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

    self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
    self.logdet = logdet

  def initialize(self, input):
    with torch.no_grad():
      flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
      mean = (
        flatten.mean(1)
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .permute(1, 0, 2, 3)
      )
      std = (
        flatten.std(1)
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .permute(1, 0, 2, 3)
      )

      self.loc.data.copy_(-mean)
      self.scale.data.copy_(1 / (std + 1e-6))

  def forward(self, input):
    _, _, height, width = input.shape

    if self.initialized.item() == 0:
      self.initialize(input)
      self.initialized.fill_(1)

    log_abs = logabs(self.scale)

    logdet = height * width * torch.sum(log_abs)

    if self.logdet:
      return self.scale * (input + self.loc), logdet

    else:
      return self.scale * (input + self.loc)

  def reverse(self, output):
    return output / self.scale - self.loc


class InvConv2d(nn.Module):
  def __init__(self, in_channel):
    super().__init__()

    weight = torch.randn(in_channel, in_channel)
    q, _ = torch.qr(weight)
    weight = q.unsqueeze(2).unsqueeze(3)
    self.weight = nn.Parameter(weight)

  def forward(self, input):
    _, _, height, width = input.shape

    out = F.conv2d(input, self.weight)
    logdet = (
      height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
    )

    return out, logdet

  def reverse(self, output):
    return F.conv2d(
      output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
    )


class InvConv2dLU(nn.Module):
  def __init__(self, in_channel):
    super().__init__()

    weight = np.random.randn(in_channel, in_channel)
    q, _ = la.qr(weight)
    w_p, w_l, w_u = la.lu(q.astype(np.float32))
    w_s = np.diag(w_u)
    w_u = np.triu(w_u, 1)
    u_mask = np.triu(np.ones_like(w_u), 1)
    l_mask = u_mask.T

    w_p = torch.from_numpy(w_p)
    w_l = torch.from_numpy(w_l)
    w_s = torch.from_numpy(w_s)
    w_u = torch.from_numpy(w_u)

    self.register_buffer('w_p', w_p)
    self.register_buffer('u_mask', torch.from_numpy(u_mask))
    self.register_buffer('l_mask', torch.from_numpy(l_mask))
    self.register_buffer('s_sign', torch.sign(w_s))
    self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
    self.w_l = nn.Parameter(w_l)
    self.w_s = nn.Parameter(logabs(w_s))
    self.w_u = nn.Parameter(w_u)

  def forward(self, input):
    _, _, height, width = input.shape

    weight = self.calc_weight()

    out = F.conv2d(input, weight)
    logdet = height * width * sum(self.w_s)

    return out, logdet

  def calc_weight(self):
    weight = (self.w_p \
      @ (self.w_l * self.l_mask + self.l_eye) \
      @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
    )

    return weight.unsqueeze(2).unsqueeze(3)

  def reverse(self, output):
    weight = self.calc_weight()

    return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
  def __init__(self, in_channel, out_channel, padding=1):
    super().__init__()

    self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
    self.conv.weight.data.zero_()
    self.conv.bias.data.zero_()
    self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

  def forward(self, input):
    out = F.pad(input, [1, 1, 1, 1], value=1)
    out = self.conv(out)
    out = out * torch.exp(self.scale * 3)

    return out

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      ActNorm(C_out, logdet=False)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      ActNorm(C_out, logdet=False),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      ActNorm(C_in, logdet=False),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      ActNorm(C_out, logdet=False).cuda(),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = ActNorm(C_out, logdet=False)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
