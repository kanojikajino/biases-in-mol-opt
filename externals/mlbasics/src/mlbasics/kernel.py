#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from gpytorch.kernels import *
import torch


class TanimotoJaccardKernel(Kernel):

    is_stationary = False

    def forward(self, x1, x2, **kwargs):
        '''

        Parameters
        ----------
        x1, x2 : torch.tensor of shape (sample_size, dim)
            rows are dense bit vectors

        Returns
        -------
        kernel : torch.tensor of shape (sample_size_1, sample_size_2)
        '''
        kernel = torch.zeros(x1.shape[0],
                             x2.shape[0],
                             dtype=torch.float32)
        for each_row_idx, each_row in enumerate(x1):
            numerator = (each_row * x2).sum(axis=1)
            denominator = torch.clip(each_row + x2, max=1.).sum(axis=1)
            kernel[each_row_idx, :] = numerator / denominator
        return kernel
