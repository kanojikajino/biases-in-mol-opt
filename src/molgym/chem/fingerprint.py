#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'


import itertools
import math
import gzip
import pickle
import sklearn.preprocessing
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.utils.validation import check_is_fitted
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
from rdkit import Chem
from .qsar import SAScore, MultiLayerPerceptronQSAR
from ..ml.feature import FeatureBase
from ..utils import mol2hash
from ..rl.env import simulate


def construct_sub_zdd(mol, dir_obj, action_space, fingerprint):
    #smiles = Chem.MolToSmiles(mol)
    file_path_obj = dir_obj / (mol2hash(mol) + '.pklz')
    if file_path_obj.exists():
        #return smiles, file_path_obj
        return None
    from graphillion import setset
    sascorer = SAScore()
    feature_list = []
    for each_action in action_space.legal_go_action_generator(
            {'current_mol': mol, 'n_step': 0}):
        product_mol, success \
            = simulate(mol,
                       each_action,
                       action_space,
                       sascorer)
        if success:
            feature = fingerprint.forward(product_mol,
                                          no_torch=True)
            feature = set(feature)
            feature_list.append(feature)
    my_setset = setset(feature_list)
    setset_size = len(my_setset)
    output = (my_setset.dumps(), my_setset.universe())
    with gzip.open(file_path_obj, 'wb') as f:
        pickle.dump(output, f)
    del setset
    del my_setset
    del output
    import gc
    gc.collect()
    #return smiles, file_path_obj
    return setset_size


def construct_sub_zdds(mol_list, dir_obj, action_space, fingerprint):
    [construct_sub_zdd(each_mol, dir_obj, action_space, fingerprint)\
     for each_mol in mol_list]
    #return [construct_sub_zdd(each_mol, dir_obj, action_space, fingerprint)\
    #        for each_mol in mol_list]


class SparseMorganFingerprint(FeatureBase):

    ''' Morgan fingerprint with ZDD data structure

    Attributes
    ----------
    radius : int
        radius of the Morgan fingerprint
    bit : int
        length of the bit vector
    '''

    def __init__(self,
                 radius=2,
                 bit=4096,
                 use_chirality=True,
                 use_zdd=True,
                 device='cpu'):
        super().__init__(device=device)
        self.radius = radius
        self.bit = bit
        self.use_chirality = use_chirality
        self.use_zdd = use_zdd
        #self.setset_dict = {}

    @property
    def is_sparse(self):
        return True

    @property
    def out_dim(self):
        return self.bit

    def forward(self, mol, no_torch=False):
        ''' return nonzero indices of Morgan fingerprint bit vector

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        torch.tensor
            each element corresponds to a non-zero index of the bit vector
        '''
        if mol is None:
            if no_torch:
                return []
            return torch.tensor([], device=self.device)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.bit,
                useChirality=self.use_chirality)
            if no_torch:
                return fp.GetOnBits()
            return torch.tensor(fp.GetOnBits(), device=self.device)

    def prep(self,
             episode_memory,
             dir_obj,
             action_space,
             logger=print,
             workers=1,
             per_batch_n_jobs=1,
             **kwargs):
        ''' prepare setset
        '''
        if not dir_obj.exists():
            dir_obj.mkdir()
        self.dir_obj = dir_obj
        self.workers = workers

        if self.use_zdd:
            next_mol_set = set(each_transition.next_state['current_mol']
                               for each_transition
                               in episode_memory.transition_list
                               if not each_transition.done)
            next_mol_set.discard(None)
            next_mol_list = list(next_mol_set)
            setset_size_list = []

            logger(' * # of tasks = {}'.format(len(next_mol_list)))
            if workers == 1:
                setset_size_list = [construct_sub_zdd(each_mol, dir_obj, action_space, self)\
                                    for each_mol in next_mol_list]
            else:
                per_batch_total_jobs = workers * per_batch_n_jobs
                n_batches = math.ceil(len(next_mol_set) / per_batch_total_jobs)
                for each_batch_idx in range(n_batches):
                    setset_size_list += Parallel(
                        n_jobs=workers,
                        verbose=10)([
                            delayed(construct_sub_zdd)(
                                each_mol,
                                dir_obj,
                                action_space,
                                self)
                            for each_mol in next_mol_list[
                                    each_batch_idx * per_batch_total_jobs \
                                    : (each_batch_idx + 1) * per_batch_total_jobs]])
                    get_reusable_executor().shutdown(wait=True)
            logger(' * total zdd size: {}'.format(sum(setset_size_list)))
        else:
            self.dir_obj = None    
        return self

    def get_zdd_path(self, mol):
        try:
            if mol:
                return self.dir_obj / (mol2hash(mol) + '.pklz')
        except:
            pass
        return None

    def zdd_exists(self, mol):
        path_obj = self.get_zdd_path(mol)
        if path_obj:
            return path_obj.exists()
        return False


class DenseMorganFingerprint(FeatureBase):

    def __init__(self,
                 radius=2,
                 bit=4096,
                 use_chirality=True,
                 device='cpu'):
        super().__init__(device=device)
        self.radius = radius
        self.bit = bit
        self.use_chirality = use_chirality
        self.device = device

    @property
    def is_sparse(self):
        return False

    @property
    def out_dim(self):
        return self.bit

    def forward(self, mol):
        if mol is None:
            return torch.tensor([], device=self.device)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.bit,
                useChirality=self.use_chirality)
            x = torch.zeros(self.bit, device=self.device)
            x[fp.GetOnBits()] = 1.0
            return x


class MolecularDescriptorsFingerprint(FeatureBase):

    def __init__(self,
                 scaler='StandardScaler',
                 scaler_kwargs={},
                 descriptor_list=None,
                 device='cpu'):
        super().__init__(device=device)
        self.scaler = getattr(sklearn.preprocessing, scaler)(**scaler_kwargs)
        if descriptor_list is None:
            self.descriptor_list = list(list(zip(*Descriptors.descList))[0])
        else:
            self.descriptor_list = descriptor_list

    @property
    def is_sparse(self):
        return False

    @property
    def out_dim(self):
        return len(self.descriptor_list)

    @property
    def descriptor_dict(self):
        descriptor_dict = dict(Descriptors.descList)
        descriptor_dict['QED'] = qed
        return descriptor_dict

    def forward(self, mol):
        try:
            check_is_fitted(self.scaler)
            return torch.tensor(
                self.scaler.transform(
                    torch.tensor(self._forward(mol, self.descriptor_list)).reshape(1, -1)).reshape(-1),
                device=self.device)
        except:
            return torch.tensor(self._forward(mol, self.descriptor_list), device=self.device)

    @staticmethod
    def _forward(mol, descriptor_list):
        descriptor_dict = dict(Descriptors.descList)
        descriptor_dict['QED'] = qed
        Chem.SanitizeMol(mol)
        return [descriptor_dict[each_desc](mol)
                for each_desc in descriptor_list]

    def batch_forward(self, mol_list, workers=-1):
        return torch.tensor(Parallel(
            n_jobs=workers)([
                delayed(self._forward)(each_mol, self.descriptor_list)
                for each_mol
                in mol_list]),
                            device=self.device)

    def fit(self, mol_list):
        X = self.batch_forward(mol_list)
        self.scaler.fit(X)
        return X

    def fit_transform(self, mol_list):
        return torch.tensor(self.scaler.fit_transform(self.batch_forward(mol_list).cpu()), device=self.device)


class SelectedMolecularDescriptorsFingerprint(MolecularDescriptorsFingerprint):

    ''' descriptors selected by Gottipati et al.
    '''

    def __init__(self, device='cpu', **kwargs):
        super().__init__(
            device=device,
            descriptor_list=['MaxEStateIndex',
                             'MinEStateIndex',
                             'MinAbsEStateIndex',
                             'QED',
                             'MolWt',
                             'FpDensityMorgan1',
                             'BalabanJ',
                             'PEOE_VSA10',
                             'PEOE_VSA11',
                             'PEOE_VSA6',
                             'PEOE_VSA7',
                             'PEOE_VSA8',
                             'PEOE_VSA9',
                             'SMR_VSA7',
                             'SlogP_VSA3',
                             'SlogP_VSA5',
                             'EState_VSA2',
                             'EState_VSA3',
                             'EState_VSA4',
                             'EState_VSA5',
                             'EState_VSA6',
                             'FractionCSP3',
                             'MolLogP',
                             'Kappa2',
                             'PEOE_VSA2',
                             'SMR_VSA5',
                             'SMR_VSA6',
                             'EState_VSA7',
                             'Chi4v',
                             'SMR_VSA10',
                             'SlogP_VSA4',
                             'SlogP_VSA6',
                             'EState_VSA8',
                             'EState_VSA9',
                             'VSA_EState9'],
            **kwargs)


class DenseNeuralFingerprint(FeatureBase):

    ''' neural fingerprint useful for target property prediction
    Morgan fingerprint -> multi-layer perceptron -> target property
    '''

    def __init__(self,
                 seed,
                 fingerprint_kwargs={'radius': 2,
                                     'bit': 1024,
                                     'use_chirality': True,
                                     'device': 'cpu'},
                 qsar_model_kwargs={'activation': 'Softplus',
                                    'activation_at_top': False,
                                    'lmbd': 0.0,
                                    'out_dim_list': [128, 128],
                                    'device': 'cpu'},
                 device='cpu'):
        if qsar_model_kwargs['activation_at_top']:
            raise ValueError
        
        super().__init__(device=device)
        self.fingerprint = DenseMorganFingerprint(**fingerprint_kwargs)
        self.qsar_model = MultiLayerPerceptronQSAR(fingerprint=self.fingerprint,
                                                   seed=seed,
                                                   **qsar_model_kwargs)

    @property
    def is_sparse(self):
        return False

    @property
    def out_dim(self):
        return self.qsar_model.predictor.predictor[-1].in_features

    def forward(self, mol):
        return self.qsar_model.forward(mol)

    def batch_forward(self, mol_list):
        return self.qsar_model.batch_forward(mol_list)

    def fit(self, mol_list, tgt_list, logger=print, **fit_kwargs):
        res = self.qsar_model.fit(mol_list, tgt_list, logger=logger, **fit_kwargs)
        self.qsar_model.predictor.predictor = self.qsar_model.predictor.predictor[:-1]
        return res
