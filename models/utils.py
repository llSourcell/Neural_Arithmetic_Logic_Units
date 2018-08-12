import torch
import torch.nn as nn


def str2act(s):
    if s is 'none':
        return None
    elif s is 'hardtanh':
        return nn.Hardtanh()
    elif s is 'sigmoid':
        return nn.Sigmoid()
    elif s is 'relu6':
        return nn.ReLU6()
    elif s is 'tanh':
        return nn.Tanh()
    elif s is 'tanhshrink':
        return nn.Tanhshrink()
    elif s is 'hardshrink':
        return nn.Hardshrink()
    elif s is 'leakyrelu':
        return nn.LeakyReLU()
    elif s is 'softshrink':
        return nn.Softshrink()
    elif s is 'softsign':
        return nn.Softsign()
    elif s is 'relu':
        return nn.ReLU()
    elif s is 'prelu':
        return nn.PReLU()
    elif s is 'softplus':
        return nn.Softplus()
    elif s is 'elu':
        return nn.ELU()
    elif s is 'selu':
        return nn.SELU()
    else:
        raise ValueError("[!] Invalid activation function.")
