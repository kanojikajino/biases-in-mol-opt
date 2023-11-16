#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

import sklearn.linear_model
import torch
from torch import nn
from covshift.weight import KernelMeanMatching
from .base import SparsePredictorBase, DensePredictorBase


class LinearSparsePredictor(SparsePredictorBase):

    def __init__(self, in_dim, device='cpu'):
        super().__init__(device=device)
        self.in_dim = in_dim
        self.emb = nn.Embedding(num_embeddings=self.in_dim,
                                embedding_dim=1)
        nn.init.zeros_(self.emb.weight)
        self.register_parameter(name='bias',
                                param=nn.Parameter(torch.tensor(0.)))
        self.to(self.device)

    def forward(self, X_sparse):
        '''

        Parameters
        ----------
        fp : torch.tensor
            tensor containing indices

        Returns
        -------
        torch.FloatTensor
        '''
        return self.emb(X_sparse).sum() + self.bias

    def fit(self,
            X_sparse_list,
            y,
            model_name,
            model_kwargs,
            kmm_kwargs,
            X_sparse_test_list=None,
            logger=print,
            **kwargs):
        X_train = self.sparse2dense(X_sparse_list)
        X_test = self.sparse2dense(X_sparse_test_list)
        clf = getattr(sklearn.linear_model, model_name)(**model_kwargs)

        kmm = KernelMeanMatching(**kmm_kwargs)
        beta_train = kmm.fit(X_train, X_test)

        try:
            clf.fit(X_train, y, beta_train)
            logger(' * r2 = {}'.format(clf.score(X_train, y, beta_train)))
        except:
            logger(' * there exists no sample_weight option, so fit the model without it')
            clf.fit(X_train, y)
            logger(' * r2 = {}'.format(clf.score(X_train, y)))

        self.emb.weight.data = torch.tensor(clf.coef_, device=self.device).reshape(-1, 1)
        self.bias.data = torch.tensor(clf.intercept_, device=self.device)
        self.register_X(X_train)
        logger(' * # of non-zero / dim = {}'.format((clf.coef_ != 0).sum() / len(clf.coef_)))
        if hasattr(clf, 'alpha_'):
            logger(' * best lmbd = {}'.format(clf.alpha_))
        logger(' * y: mean, std, max, mean = {}, {}, {}, {}'.format(
            y.mean(),
            y.std(),
            y.max(),
            y.min()))
        pred = torch.tensor(clf.predict(X_train).reshape(-1), device=self.device)
        logger(' * pred: mean, std, max, mean = {}, {}, {}, {}'.format(
            pred.mean(),
            pred.std(),
            pred.max(),
            pred.min()))
        y = y.to(self.device)
        return 0.5 * (torch.norm(pred - y.reshape(-1)) ** 2)

    def register_X(self, X):
        self.train_X = X


class LinearDensePredictor(DensePredictorBase):

    def __init__(self, in_dim, model_name, model_kwargs, device='cpu'):
        super().__init__(device=device)
        self.clf = getattr(sklearn.linear_model, model_name)(**model_kwargs)

    def fit(self, X, y, **kwargs):
        return self.clf.fit(X, y)

    def forward(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if torch.is_tensor(X):
            return torch.Tensor(self.clf.predict(X.to('cpu').numpy()), device=self.device)
        return self.clf.predict(X)
