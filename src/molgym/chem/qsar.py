#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from abc import abstractmethod
import gzip
import os
import pickle
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.Descriptors import MolLogP
import torch
from torch import nn
from apollo1060.smol import SMol
from .sascorer import synthetic_accessibility, synthetic_accessibility_batch
from mlbasics.nn import MultiLayerPerceptron
from ..ml.gp import ExactGPModel, SklearnGPModel
from ..ml.linear import LinearDensePredictor


# ------- base classes --------

class QSARBase(nn.Module):

    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.device = device

    @abstractmethod
    def forward(self, mol):
        pass

    def fit(self, mol_list, tgt_list, **kwargs):
        return -1

    def to(self, device):
        super().to(device)
        self.device = device

class FingerprintQSARBase(QSARBase):

    ''' QSAR modeling based on fingerprint

    Attributes
    ----------
    fingerprint : Fingerprint object
        convert Mol object to vector
    '''

    def __init__(self, fingerprint, predictor, device='cpu'):
        super().__init__(device=device)
        self.fingerprint = fingerprint
        self.predictor = predictor
        if self.fingerprint.is_sparse != self.predictor.is_sparse:
            raise ValueError('fingerprint and predictor must be consistent in sparse/dense representations')
        if self.predictor.device != self.device:
            raise ValueError('predictor is located on {}, which is not consistent with device {}'.format(
                self.predictor.device,
                self.device))

    def construct_dataset(self, mol_list, tgt_list, X=None):
        if X is None:
            X = self.fingerprint.batch_forward(mol_list)
        if not self.fingerprint.is_sparse:
            X = X.to('cpu')
        y = torch.tensor(tgt_list).to('cpu')
        return X, y

    def forward(self, mol):
        return self.predictor.forward(self.fingerprint.forward(mol))

    def batch_forward(self, mol_list):
        return [self.predictor.forward(self.fingerprint.forward(each_mol)) for each_mol in mol_list]

    def delete_rng(self):
        try:
            self.predictor.delete_rng()
        except:
            pass

    def to(self, device):
        super().to(device)
        self.fingerprint.to(device)
        self.predictor.to(device)
        self.device = device


# ------- simulation (no learning) ---------

class SAScore(QSARBase):

    ''' Synthetic accessibility score
    The lower, the more accessible.
    '''

    @property
    def fscores(self):
        if not hasattr(self, '_fscores'):
            with gzip.open(os.path.join(os.path.dirname(__file__), 'fpscores.pkl.gz'), 'rb') as f:
                self._fscores = pickle.load(f)
            out_dict = {}
            for each_list in self._fscores:
                for each_idx in range(1,len(each_list)):
                    out_dict[each_list[each_idx]] = float(each_list[0])
            self._fscores = out_dict
        return self._fscores

    def forward(self, mol):
        return synthetic_accessibility(mol, self.fscores)

    def batch_forward(self, mol_list):
        return synthetic_accessibility_batch(mol_list, _fscores=self.fscores)


class SAScoreMixin:

    @property
    def sascorer(self):
        if not hasattr(self, '_sascorer'):
            self._sascorer = SAScore()
        return self._sascorer


class PenalizedLogP(SAScore):

    ''' score to be maximized
    log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    '''

    def forward(self, mol):
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logp_mean = 2.4570953396190123
        logp_std = 1.434324401111988
        sa_mean = -3.0525811293166134
        sa_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        log_p = MolLogP(mol)
        sa = - synthetic_accessibility(mol, self.fscores)
        cycle_score = - self.cycle_length(mol)

        normalized_log_p = (log_p - logp_mean) / logp_std
        normalized_sa = (sa - sa_mean) / sa_std
        normalized_cycle = (cycle_score - cycle_mean) / cycle_std

        return normalized_log_p + normalized_sa + normalized_cycle

    @staticmethod
    def cycle_length(mol):
        ''' calculate cycle score

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        int : cycle score
        '''
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))

        if not cycle_list:
            cycle_len = 0
        else:
            cycle_len = max([len(each_cycle) for each_cycle in cycle_list])
        if cycle_len <= 6:
            cycle_len = 0
        else:
            cycle_len = cycle_len - 6
        return cycle_len


class CCR5(QSARBase):

    ''' predictor developed by 99andBeyond.
    should be used for evaluation.
    '''

    smol = SMol

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe

    def forward(self, mol):
        return self.pipe.predict_vector(self.get_feature(mol))

    def get_feature(self, mol):
        smiles = Chem.MolToSmiles(mol)
        smol = self.smol(smiles)
        smol.featurize(self.pipe.features)
        return smol.features_values

    @property
    def pipe(self):
        if hasattr(self, '_pipe'):
            pass
        else:
            from apollo1060.predictors import ccr5_pipe
            self._pipe = ccr5_pipe
        return self._pipe


class HIVInt(CCR5):

    @property
    def pipe(self):
        if hasattr(self, '_pipe'):
            pass
        else:
            from apollo1060.predictors import int_pipe
            self._pipe = int_pipe
        return self._pipe


class HIVRT(CCR5):

    @property
    def pipe(self):
        if hasattr(self, '_pipe'):
            pass
        else:
            from apollo1060.predictors import rt_pipe
            self._pipe = rt_pipe
        return self._pipe


# ------ learning -------

class LinearDenseQSAR(FingerprintQSARBase):

    def __init__(self,
                 fingerprint,
                 model_name,
                 model_kwargs,
                 device='cpu',
                 **kwargs):
        predictor = LinearDensePredictor(fingerprint.out_dim,
                                         model_name,
                                         model_kwargs,
                                         device=device)
        super().__init__(fingerprint, predictor, device=device)

    def fit(self, mol_list, tgt_list, **kwargs):
        try:
            X = self.fingerprint.fit(mol_list)
        except:
            X = None

        X, y = self.construct_dataset(
            mol_list,
            tgt_list,
            X=X)
        return self.predictor.fit(X, y)

    def batch_forward(self, mol_list):
        X = torch.stack([self.fingerprint.forward(each_mol) for each_mol in mol_list])
        return self.predictor.forward(X)


class MultiLayerPerceptronQSAR(FingerprintQSARBase):

    def __init__(self,
                 fingerprint,
                 seed,
                 activation='ReLU',
                 out_dim_list=[512, 512],
                 lmbd=1e-2,
                 activation_at_top=False,
                 device='cpu',
                 **kwargs):
        in_dim = fingerprint.out_dim
        predictor = MultiLayerPerceptron(
            in_dim=in_dim,
            seed=seed,
            activation=activation,
            out_dim_list=out_dim_list,
            out_dim=1,
            lmbd=lmbd,
            activation_at_top=activation_at_top,
            device=device)
        super().__init__(fingerprint, predictor, device=device)

    def fit(self,
            mol_list,
            tgt_list,
            batch_size=2,
            n_epochs=100,
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-1},
            print_freq=10,
            logger=print,
            **kwargs):
        try:
            X = self.fingerprint.fit(mol_list)
        except:
            X = None
        X, y = self.construct_dataset(
            mol_list,
            tgt_list,
            X=X)
        return self.predictor.fit(
            X,
            y,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            print_freq=print_freq,
            logger=logger,
            **kwargs)

    def batch_forward(self, mol_list):
        X = torch.stack([self.fingerprint.forward(each_mol) for each_mol in mol_list])
        return self.predictor.forward(X)


class ExactGPQSAR(FingerprintQSARBase):

    def __init__(self, fingerprint, covar_module_name, device='cpu', **kwargs):
        predictor = ExactGPModel(covar_module_name)
        super().__init__(fingerprint, predictor, device=device)

    def fit(self,
            mol_list,
            tgt_list,
            n_epochs=5,
            optimizer_kwargs={'lr': 1e-2},
            logger=print):
        try:
            X = self.fingerprint.fit(mol_list)
        except:
            X = None

        X, y = self.construct_dataset(
            mol_list,
            tgt_list,
            X=X)
        return self.predictor.fit(
            X.to(torch.float32),
            y.to(torch.float32),
            n_epochs,
            optimizer_kwargs,
            logger)

    def batch_forward(self, mol_list):
        X = torch.stack([self.fingerprint.forward(each_mol) for each_mol in mol_list]).to(torch.float32)
        return self.predictor.forward(X)


class SklearnGPQSAR(FingerprintQSARBase):

    def __init__(self, fingerprint, **kwargs):
        predictor = SklearnGPModel()
        super().__init__(fingerprint, predictor)

    def fit(self, mol_list, tgt_list, **kwargs):
        try:
            X = self.fingerprint.fit(mol_list)
        except:
            X = None
        X, y = self.construct_dataset(mol_list, tgt_list, X=X)
        self.predictor.fit(X.numpy(), y.numpy())
