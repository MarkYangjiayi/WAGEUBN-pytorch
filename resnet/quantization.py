import torch
import torch.nn as nn

bitsW = 8
bitsA = 8
bitsE = 8
bitsG = 8
bitsU = 24

def scale(x):
    result = 2.**torch.round(torch.log2(x))
    return result

def del(bits):
    result = (2.**(1-bits))
    return result

def clip(x, bits):
    if bits >= 32:
        delta = 0
    else:
        delta = del(bits)
    ceil  = 1 - delta
    floor = delta - 1
    result = torch.clamp(x, floor, ceil)
    return result

def Q(x, bits):
    if bits >= 32:
        result = x
    else:
        result = torch.round(x/del(bits))*del(bits)
    return result

def QW(x):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = clip(Q(x,bits),bits)
    return

def QA(x):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        result = clip(Q(x,bits),bits)
    return

def QE(x):
    bits = bitsE
    if bits >= 32:
        result = x
    else:
        result = clip(Q(x,bits),bits)
    return

def QG(x):
    bits = bitsG
    if bits >= 32:
        result = x
    else:
        result = clip(Q(x,bits),bits)
    return

def QU(x):
    bits = bitsU
    if bits >= 32:
        result = x
    else:
        result = clip(Q(x,bits),bits)
    return
