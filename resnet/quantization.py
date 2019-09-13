import torch
import torch.nn as nn
from torch.autograd import Function

bitsW = 8
bitsA = 8
bitsE = 8
bitsG = 8
bitsU = 16

# bitsW = 8
# bitsA = 8
# bitsE = 8
# bitsG = 8
# bitsU = 24

def scale(x):
    scale = torch.max(torch.abs(x))
    result = 2.**torch.round(torch.log2(scale))
    return result

def delta(bits):
    result = (2.**(1-bits))
    return result

def clip(x, bits):
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil  = 1 - step
    floor = step - 1
    result = torch.clamp(x, floor, ceil)
    return result

def quant(x, bits):
    if bits >= 32:
        result = x
    else:
        result = torch.round(x/delta(bits))*delta(bits)
    return result

def qw(x):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = clip(quant(x,bits),bits)
    return result

def qa(x):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        result = quant(x,bits)
    return result

def qe(x):
    bits = bitsE
    if bits >= 32:
        result = x
    else:
        dscale = scale(x)
        result = dscale*clip(quant(x/dscale,bits),bits)
    return result

def qg(x):
    bits = bitsG
    if bits >= 32:
        result = x
    else:
        # dscale = scale(x)
        # x = x / dscale
        # factor = 128
        # bitsR = 32
        # norm = quant(factor * x, bitsR)
        #
        # norm_sign = torch.sign(norm)
        # norm_abs = torch.abs(norm)
        # norm_int = torch.floor(norm_abs)
        # norm_float = norm_abs - norm_int
        # rand_float = torch.FloatTensor(*x.size()).uniform_()
        # norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )
        # norm = torch.clamp(norm,-factor+1,factor-1)
        # result = quant(norm*delta(bits)/128,15)

        dscale = scale(x)
        x = x / dscale
        factor = 128
        bitsR = 32
        norm = quant(factor * x, bitsR)

        norm_sign = torch.sign(norm)
        norm_abs = torch.abs(norm)
        norm_int = torch.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = torch.FloatTensor(*x.size()).uniform_()
        norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )
        norm = torch.clamp(norm,-factor+1,factor-1)
        result = norm/128

    return result

def qu(x):
    bits = bitsU
    if bits >= 32:
        result = x
    else:
        result = clip(quant(x,bits),bits)
    return result

class QW(Function):
    @staticmethod
    def forward(self, x):
        result = qw(x)
        return result

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input
quantizeW = QW().apply

class QAE(Function):
    @staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        result = qa(x)
        return result

    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_input = qe(grad_output)
        else:
            grad_input = grad_output
        return grad_input
quantizeAE = QAE.apply

import numpy as np
np.random.seed(10)
shape = (5,5)
test_data = np.random.rand(*shape)
test_tensor = torch.from_numpy(test_data).float()
result = qg(test_tensor)
print(test_tensor)
print(result)
