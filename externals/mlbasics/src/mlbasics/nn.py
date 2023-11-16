#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

import math
import torch
from torch import nn
from .base import DensePredictorBase
from .utils import TorchRngMixin
from torch.utils.data import TensorDataset, DataLoader


class MultiLayerPerceptron(TorchRngMixin, DensePredictorBase):

    def __init__(self,
                 in_dim,
                 seed,
                 activation='Softplus',
                 activation_kwargs={},
                 out_dim_list=[512, 512, 1],
                 out_dim=None,
                 lmbd=0.0,
                 activation_at_top=False,
                 top_activation=None,
                 top_activation_kwargs={},
                 device='cpu',
                 **kwargs):
        super().__init__(device=device)
        self.set_torch_seed(seed)
        torch.manual_seed(self.gen_seed())
        self.device = device
        self.in_dim = in_dim
        self.lmbd = lmbd

        out_dim_list = list(out_dim_list)
        #in_dim = 0
        if out_dim is not None:
            out_dim_list.append(out_dim)
        module_list = []

        activation_list = []
        if isinstance(activation, str):
            for _ in out_dim_list:
                activation_list.append(activation)
        else:
            activation_list = list(activation)

        out_dim = in_dim
        for each_out_dim, each_activation in zip(out_dim_list, activation_list):
            in_dim = out_dim
            out_dim = each_out_dim
            module_list.append(nn.Linear(in_dim, out_dim))
            module_list.append(getattr(nn, each_activation)(**activation_kwargs))
        module_list.pop()
        if activation_at_top:
            if top_activation is None:
                module_list.append(getattr(nn, each_activation)(**activation_kwargs))
            else:
                module_list.append(getattr(nn, top_activation)(**top_activation_kwargs))
        self.predictor = nn.Sequential(*module_list)
        self.to(self.device)
        self.dtype = next(self.predictor.parameters()).dtype
        self.delete_rng()

    def reset_parameters(self):
        def _reset(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.predictor.apply(_reset)

    def forward(self, X):
        #device = list(self.parameters())[0].device
        X = X.to(self.device, self.dtype)
        return self.predictor.forward(X)

    def fit(self,
            X,
            y,
            weight=None,
            batch_size=1,
            n_epochs=100,
            weight_decay=1.0,
            max_update=None,
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
            print_freq=10,
            logger=print):
        assert len(X) == len(y)
        #device = list(self.parameters())[0].device
        sample_size = len(X)
        if weight is not None:
            weight = weight ** weight_decay
        '''
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            generator=self.torch_rng_cpu)
        '''
        optimizer = getattr(torch.optim, optimizer)(
            params=self.parameters(), **optimizer_kwargs)

        if max_update is not None:
            n_epochs = math.ceil(batch_size * max_update / sample_size)
            logger(' * n_epochs is changed to {}'.format(n_epochs))
        else:
            max_update = n_epochs * sample_size
        # training
        for iter_idx in range(max_update):
            running_loss = 0
            counter = 0
            idx_tensor = torch.randperm(sample_size,
                                        generator=self.torch_rng_cpu)[:batch_size]
            #idx_tensor = torch.randint(high=sample_size,
            #                           size=(batch_size,),
            #                           generator=self.torch_rng_cpu)
            each_X = X[idx_tensor]
            each_y = y[idx_tensor]
            if weight is not None:
                each_weight = weight[idx_tensor]
            else:
                each_weight = None
            loss = self.loss(each_X,
                             each_y,
                             weight=each_weight,
                             with_regularizer=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter += 1
            if print_freq > 0 and iter_idx % print_freq == 0:
                logger('#(update) = {}\t loss = {}'.format(
                    iter_idx,
                    running_loss/counter))
        return running_loss/counter

    def loss(self, X, y, weight=None, with_regularizer=True):
        ''' compute loss function
        '''
        #device = list(self.parameters())[0].device
        batch_size = len(X)
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype).reshape(-1, 1)
        if weight is None:
            loss = 0.5 * ((y - self.forward(X)) ** 2).mean()
        else:
            loss = 0.5 * (weight @ (y - self.forward(X)) ** 2) / batch_size
        if with_regularizer:
            loss = loss + 0.5 * self.lmbd * sum([torch.norm(each_param) ** 2
                                                 for each_param
                                                 in self.parameters()])
        return loss
