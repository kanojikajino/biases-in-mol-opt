#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' feature representation of state-action pair
'''

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from abc import abstractmethod
import gzip
import pickle
from rdkit import Chem
import torch
from ..ml.feature import FeatureBase
from ..chem.qsar import SAScoreMixin
from .env import simulate


# base class

class StateFeatureBase(FeatureBase, SAScoreMixin):

    ''' base class of a feature extraction module,
    which maps an observation into some generic formats.
    '''

    def __init__(self, fingerprint, device='cpu'):
        super().__init__(device=device)
        if fingerprint.device != device:
            raise ValueError('fingerprint and state_feature must be in the same device.')
        self.fingerprint = fingerprint

    @abstractmethod
    def forward(self, observation):
        ''' action_space is in the argument because it consumes too much memory.
        '''
        raise NotImplementedError


class StateActionFeatureBase(FeatureBase, SAScoreMixin):

    ''' base class of a feature extraction module,
    which maps an observation into some generic formats.
    '''

    def __init__(self, fingerprint, device='cpu', **kwargs):
        super().__init__(device=device)
        if fingerprint.device != device:
            raise ValueError('fingerprint and state_action_feature must be in the same device.')
        self.fingerprint = fingerprint

    @abstractmethod
    def forward(self, observation, action, action_space):
        ''' action_space is in the argument because it consumes too much memory.
        '''
        raise NotImplementedError

    @abstractmethod
    def stop_forward(self, observation):
        ''' when action is `stop`
        '''
        raise NotImplementedError

    def batch_forward(self, obs_action_list, action_space):
        forward_list = [self.forward(each_obs, each_action, action_space)
                        for each_obs, each_action
                        in obs_action_list]
        if isinstance(forward_list[0], list) or self.is_sparse:
            return forward_list
        return torch.stack(forward_list)


# mixin

class Sparse2DenseMixin:

    @property
    def is_sparse(self):
        return False

    def forward(self, **kwargs):
        idx_tensor = super().forward(**kwargs)
        x = torch.zeros(self.out_dim, device=self.device)
        x[idx_tensor] = 1.0
        return x

    def stop_forward(self, **kwargs):
        idx_tensor = super().stop_forward(**kwargs)
        x = torch.zeros(self.out_dim, device=self.device)
        x[idx_tensor] = 1.0
        return x


# specific base classes

class MolStateFeatureBase(StateFeatureBase):

    def forward(self, observation):
        return self._forward(current_mol=observation['current_mol'],
                             n_step=observation['n_step'])

    @abstractmethod
    def _forward(self, current_mol, n_step):
        raise NotImplementedError


class MolAndStepNumberObservationFeatureBase(StateActionFeatureBase):

    ''' observation = dict[mol, n_step]
    '''

    def forward(self, observation, action, action_space):
        return self._forward(current_mol=observation['current_mol'],
                             n_step=observation['n_step'],
                             action=action,
                             action_space=action_space)

    def stop_forward(self, observation):
        return self._stop_forward(current_mol=observation['current_mol'],
                                  n_step=observation['n_step'])

    @abstractmethod
    def _forward(self, current_mol, n_step, action, action_space):
        raise NotImplementedError

    @abstractmethod
    def _stop_forward(self, current_mol, n_step):
        raise NotImplementedError

    def obs2key(self, observation):
        return observation['current_mol']


# def

class SparseMorganFingerprintStateFeature(MolStateFeatureBase):

    @property
    def is_sparse(self):
        return True

    @property
    def out_dim(self):
        return self.fingerprint.out_dim

    def _forward(self, current_mol, n_step):
        return self.fingerprint.forward(current_mol)


class DenseMorganFingerprintStateFeature(Sparse2DenseMixin,
                                         SparseMorganFingerprintStateFeature):
    pass


class SingleSparseFingerprintFeature(MolAndStepNumberObservationFeatureBase):

    @property
    def is_sparse(self):
        return True

    @property
    def out_dim(self):
        return self.fingerprint.out_dim

    def _forward(self, current_mol, n_step, action, action_space):
        ''' get feature vector of (s, a).
        '''
        feature_list = []
        product_mol, success = simulate(current_mol,
                                        action,
                                        action_space,
                                        self.sascorer)
        if not success:
            raise ValueError('action {} is not feasible'.format(action))
        feature_list.append(self.fingerprint.forward(product_mol))
        return torch.cat(feature_list, axis=0).sort().values

    def _stop_forward(self, current_mol, n_step):
        ''' get feature vector of (s, a) where a is the stop action
        '''
        feature_list = []
        fp = self.fingerprint.forward(current_mol)
        feature_list.append(fp)
        return torch.cat(feature_list, axis=0).sort().values

    @staticmethod
    def light_greedy_predict(observation,
                             serialized_setset_path,
                             linear_sparse_predictor,
                             fingerprint):
        '''
        ** for parallel computing using ZDD **
        self.setset_dict is too large for multiprocessing
        '''
        current_mol = observation['current_mol']
        from graphillion import setset
        if current_mol is None or serialized_setset_path is None:
            return torch.tensor([0.])
        with torch.no_grad():
            try:
                with gzip.open(serialized_setset_path, 'rb') as f:
                    serialized_setset_tuple = pickle.load(f)
            except:
                print('current_mol: {}'.format(Chem.MolToSmiles(current_mol)))
                print(observation)
                raise ValueError
            zdd = setset(setset.loads(serialized_setset_tuple[0]))
            zdd.set_universe(serialized_setset_tuple[1])                
            if len(zdd) >= 1:
                idx_list = torch.tensor(zdd.universe())
                #idx_list = idx_list[idx_list < bit]
                weights = dict(zip(
                    idx_list.tolist(),
                    linear_sparse_predictor.emb.weight.reshape(-1)[idx_list].tolist()))
                argmax_fp = list(next(zdd.max_iter(weights=weights)))
                tensor_fp = torch.tensor(sorted(argmax_fp))
                _q_max = linear_sparse_predictor.forward(tensor_fp)
                return _q_max
            else:
                current_fp = fingerprint.forward(current_mol)
                return linear_sparse_predictor.forward(torch.tensor(
                    sorted(current_fp.tolist())))



class SingleDenseFingerprintFeature(Sparse2DenseMixin, SingleSparseFingerprintFeature):
    pass


class DoubleSparseFingerprintFeature(MolAndStepNumberObservationFeatureBase):

    @property
    def is_sparse(self):
        return True

    @property
    def out_dim(self):
        return self.fingerprint.out_dim * 2

    def _forward(self, current_mol, n_step, action, action_space):
        ''' get feature vector of (s, a).
        '''
        feature_list = []
        feature_list.append(self.fingerprint.forward(current_mol) + self.fingerprint.out_dim)

        product_mol, success = simulate(current_mol,
                                        action,
                                        action_space,
                                        self.sascorer)
        if not success:
            raise ValueError('action {} is not feasible'.format(action))
        feature_list.append(self.fingerprint.forward(product_mol))
        return torch.cat(feature_list, axis=0).sort().values

    def _stop_forward(self, current_mol, n_step):
        ''' get feature vector of (s, a) where a is the stop action
        '''
        feature_list = []
        fp = self.fingerprint.forward(current_mol)
        feature_list.append(fp + self.fingerprint.out_dim)
        feature_list.append(fp)
        return torch.cat(feature_list, axis=0).sort().values

    @staticmethod
    def light_greedy_predict(observation,
                             serialized_setset_path,
                             linear_sparse_predictor,
                             fingerprint):
        '''
        ** for parallel computing using ZDD **
        self.setset_dict is too large for multiprocessing
        '''
        current_mol = observation['current_mol']
        from graphillion import setset
        if current_mol is None or serialized_setset_path is None:
            return torch.tensor([0.])
        with torch.no_grad():
            with gzip.open(serialized_setset_path, 'rb') as f:
                serialized_setset_tuple = pickle.load(f)

            zdd = setset(setset.loads(serialized_setset_tuple[0]))
            zdd.set_universe(serialized_setset_tuple[1])
            current_fp = fingerprint.forward(current_mol)
            if len(zdd) >= 1:
                idx_list = torch.tensor(zdd.universe())
                idx_list = idx_list[idx_list < fingerprint.out_dim]
                weights = dict(zip(
                    idx_list.tolist(),
                    linear_sparse_predictor.emb.weight.reshape(-1)[idx_list].tolist()))
                argmax_fp = list(next(zdd.max_iter(weights=weights)))
                tensor_fp = torch.tensor(sorted(
                    argmax_fp + (current_fp + fingerprint.out_dim).tolist()))
                _q_max = linear_sparse_predictor.forward(tensor_fp)
                return _q_max
            else:
                return linear_sparse_predictor.forward(torch.tensor(
                    sorted((current_fp + fingerprint.out_dim).tolist()
                           + current_fp.tolist())))



class DoubleDenseFingerprintFeature(Sparse2DenseMixin, DoubleSparseFingerprintFeature):
    pass



class ForwardSynthesisStateActionFeature(MolAndStepNumberObservationFeatureBase):

    ''' observation = dict[mol, n_step]
    '''

    def __init__(self, fingerprint, action_space, device='cpu'):
        super().__init__(fingerprint=fingerprint, device=device)
        self._out_dim = fingerprint.out_dim + action_space.out_dim
        self.action_space_out_dim = action_space.out_dim

    @property
    def is_sparse(self):
        return False

    @property
    def out_dim(self):
        return self._out_dim

    def _forward(self, current_mol, n_step, action, action_space):
        mol_array = self.fingerprint.forward(current_mol)
        if action[0] is not None:
            template_action = action[0].to(self.device)
            reactant_action = action[1].to(self.device)
            return torch.cat([mol_array,
                              template_action,
                              reactant_action])
        return self._stop_forward(current_mol, n_step)

    def _stop_forward(self, current_mol, n_step):
        mol_array = self.fingerprint.forward(current_mol)
        action_array = torch.zeros(self.out_dim - self.fingerprint.out_dim,
                                   device=self.device)
        return torch.cat([mol_array, action_array])


class ForwardSynthesisStateDiscreteActionFeature(MolAndStepNumberObservationFeatureBase):

    ''' observation = dict[mol, n_step]
    '''

    def __init__(self, fingerprint, action_space, device='cpu'):
        super().__init__(fingerprint=fingerprint, device=device)
        self._out_dim = fingerprint.out_dim + action_space.out_dim
        self.action_space_out_dim = action_space.out_dim

    @property
    def is_sparse(self):
        return False

    @property
    def out_dim(self):
        return self._out_dim

    def _forward(self, current_mol, n_step, action, action_space):
        mol_array = self.fingerprint.forward(current_mol)
        if action[0] is not None and isinstance(action[0], int):
            template_action = torch.zeros(action_space.reaction_corpus.n_reaction, device=self.device)
            template_action[action[0]] = 1.0
            reactant_action = torch.zeros(action_space.reaction_corpus.n_reactant, device=self.device)
            reactant_action[action[1]] = 1.0
            return torch.cat([mol_array,
                              template_action,
                              reactant_action])
        elif action[0] is not None and not isinstance(action[0], int):
            return torch.cat([mol_array,
                              action[0],
                              action[1]])
        else:
            pass
        return self._stop_forward(current_mol, n_step)

    def _stop_forward(self, current_mol, n_step):
        mol_array = self.fingerprint.forward(current_mol)
        action_array = torch.zeros(self.out_dim - self.fingerprint.out_dim,
                                   device=self.device)
        return torch.cat([mol_array, action_array])
