#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from abc import abstractmethod
from torch import nn
import torch


class DensePredictorBase(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    @property
    def is_sparse(self):
        return False

    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    def forward_std(self, X):
        X.to(self.device)
        return torch.zeros(X.shape[0], device=self.device)

    def forward_pessimistic(self, X, pessimism=1.0):
        return self.forward(X) - pessimism * self.forward_std(X)

    def batch_forward(self, X):
        return self.forward(X)

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        self.device = device


class SparsePredictorBase(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    @property
    def is_sparse(self):
        return True

    @abstractmethod
    def fit(self, X_sparse_list, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, X_sparse) -> torch.tensor:
        raise NotImplementedError

    def forward_std(self, X_sparse):
        return torch.zeros(len(X_sparse), device=self.device)

    def forward_pessimistic(self, X_sparse, pessimism=1.0):
        return self.forward(X_sparse) \
            - pessimism * self.forward_std(X_sparse)

    def sparse2dense(self, X_sparse_list):
        X = torch.zeros((len(X_sparse_list), self.in_dim), device=self.device)
        for each_idx, each_x_sparse in enumerate(X_sparse_list):
            X[each_idx, each_x_sparse] = 1.0
        return X

    def batch_forward(self, X_sparse_list):
        return torch.cat([self.forward(each_X) for each_X in X_sparse_list])

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        self.device = device


class EnsemblePredictorListBase(nn.Module):

    def __init__(self, module_list, device):
        super().__init__()
        self.module_list = nn.ModuleList(module_list)
        self.device = device
        self.to(self.device)

    def append(self, module):
        self.module_list.append(module)

    def forward(self, X):
        pred = 0
        for each_module in self.module_list:
            pred = pred + each_module.forward(X)
        return pred / len(self.module_list)

    def batch_forward(self, X):
        pred = 0
        for each_module in self.module_list:
            pred = pred + each_module.batch_forward(X)
        return pred / len(self.module_list)
