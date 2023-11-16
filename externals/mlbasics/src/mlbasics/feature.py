#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

import torch
from torch import nn


# base class

class FeatureBase(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    @property
    def is_sparse(self):
        raise NotImplementedError

    @property
    def out_dim(self):
        raise NotImplementedError

    def forward(x, **kwargs):
        raise NotImplementedError

    def batch_forward(self, x_iter, **kwargs):
        forward_list = [self.forward(each_x, **kwargs) for each_x in x_iter]
        if isinstance(forward_list[0], list) or self.is_sparse:
            return forward_list
        return torch.stack(forward_list)

    def to(self, device):
        self.device = device
        return self
