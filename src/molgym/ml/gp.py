#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import molgym.ml.kernel
from mlbasics.base import DensePredictorBase


class ExactGPModel(DensePredictorBase):

    class _ExactGPModel(ExactGP):

        def __init__(self, train_x, train_y, likelihood, covar_module):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = covar_module

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    def __init__(self, covar_module_name, device='cpu'):
        super().__init__(device=device)
        self.likelihood = GaussianLikelihood()
        self.gp_model = None
        self.covar_module = ScaleKernel(getattr(molgym.ml.kernel, covar_module_name)())

    def fit(self,
            X,
            y,
            n_epochs=10,
            optimizer_kwargs={'lr': 1e-1},
            logger=print,
            print_freq=10,
            **kwargs):
        X = X.to(self.device)
        y = y.to(self.device)
        self.gp_model = self._ExactGPModel(X,
                                           y,
                                           self.likelihood,
                                           self.covar_module)
        self.gp_model.to(self.device)
        self.gp_model.train()
        self.likelihood.to(self.device)
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), **optimizer_kwargs)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        for i in range(n_epochs):
            optimizer.zero_grad()
            pred_y = self.gp_model(X)
            loss = -mll(pred_y, y)
            loss.backward()
            if i % print_freq == 0:
                logger('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, n_epochs, loss.item(),
                    0.,
                    self.gp_model.likelihood.noise.item()
                ))
            optimizer.step()
        self.gp_model.eval()
        self.likelihood.eval()
        return self

    def forward(self, X):
        X = X.to(self.device)
        f_pred = self.gp_model(X)
        return f_pred.mean

    def forward_std(self, X):
        f_pred = self.gp_model(X)
        X = X.to(self.device)
        return torch.sqrt(f_pred.variance)


class SklearnGPModel(DensePredictorBase):

    def __init__(self):
        super().__init__(device='cpu')
        self.gp = GaussianProcessRegressor()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.gp = self.gp.fit(self.X, self.y)

    def forward(self, X) -> torch.Tensor:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return torch.tensor(self.gp.predict(X))
