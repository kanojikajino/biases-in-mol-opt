#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2021'

from abc import abstractmethod
import gzip
import pickle
import torch
from torch import nn
from rdkit import Chem
from mlbasics.nn import MultiLayerPerceptron
from mlbasics.utils import OptimizerMixin
from ..chem.qsar import SAScoreMixin
from ..ml.linear import LinearSparsePredictor


class ActionValueFunctionBase(OptimizerMixin, nn.Module):

    ''' Base class of action value function.
    `predict` must be implemented.
    '''

    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.device = device
        self.optimizer = None

    def to(self, device):
        self.device = device
        super().to(device)

    @abstractmethod
    def reset_parameters(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, observation, action, action_space):
        ''' return Q(s, a)
        '''
        raise NotImplementedError

    @abstractmethod
    def stop_forward(self, observation):
        ''' return Q(s, a) where a is the stop action
        '''
        raise NotImplementedError

    def batch_forward(self, obs_action_list, action_space):
        forward_list = [self.forward(each_obs, each_action, action_space)
                        for each_obs, each_action
                        in obs_action_list]
        return torch.cat(forward_list)

    def greedy_forward(self, observation, action_space):
        ''' return (max_a Q(s, a), argmax_a Q(s, a))
        '''
        q_max = torch.tensor(-float('inf'), device=self.device)
        best_action = None
        for each_action in action_space.legal_go_action_generator(observation):
            try:
                q_val = self.forward(observation,
                                     each_action,
                                     action_space)
                if q_max.item() < q_val.item():
                    q_max = q_val
                    best_action = each_action
            except ValueError:
                pass
        q_stop = self.stop_forward(observation)
        if q_max.item() < q_stop.item():
            q_max = q_stop
            best_action = action_space.stop_action
        return q_max, best_action

    def greedy_predict(self, observation, action_space):
        ''' return max_a Q(s, a)
        '''
        return self.greedy_forward(observation, action_space)[0]

    def greedy_action(self, observation, action_space):
        ''' return argmax_a Q(s, a)
        '''
        return self.greedy_forward(observation, action_space)[1]

    def fit(self,
            state_action_train_list,
            y_train,
            action_space,
            n_epochs=10,
            **kwargs):
        for _ in range(n_epochs):
            loss = self.loss(state_action_train_list, y_train, action_space)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def loss(self, state_action_train_list, y_train, action_space):
        loss = 0
        y_pred = self.batch_forward(state_action_train_list, action_space).reshape(-1)
        y_train = y_train.to(self.device)
        return ((y_pred - y_train) ** 2).sum() / len(y_train)


class ZeroActionValueFunction(ActionValueFunctionBase):

    ''' Always predict zero
    '''

    def forward(self, observation, action, action_space):
        return torch.tensor([0.], device=self.device)

    def greedy_forward(self, observation, action_space):
        action = next(action_space.legal_go_action_generator(observation))
        return torch.tensor([0.], device=self.device), action


class TwoStageActionValueFunction(ActionValueFunctionBase, SAScoreMixin):

    def __init__(self,
                 predictor,
                 state_action_feature,
                 device='cpu'):
        super().__init__(device=device)
        if predictor.device != device or state_action_feature.device != device:
            raise ValueError('predictor, state_action_feature, '\
                             'and action value function must be in the same device.')
        if predictor.in_dim != state_action_feature.out_dim:
            raise ValueError('predictor\'s in_dim is inconsistent with fingerprint.')
        if predictor.is_sparse != state_action_feature.is_sparse:
            raise ValueError('predictor and state_action_feature representation'\
                             ' must be consistent in terms of sparse/dense.')


        self.predictor = predictor
        self.state_action_feature = state_action_feature

    @property
    def feature_dim(self):
        return self.state_action_feature.out_dim

    @property
    def stop_predictor(self):
        if hasattr(self, '__stop_predictor'):
            return self.__stop_predictor
        return self.predictor

    @stop_predictor.setter
    def stop_predictor(self, predictor):
        self.__stop_predictor = predictor

    def forward(self, observation, action, action_space):
        return self.predictor.forward(
            self.state_action_feature.forward(observation,
                                              action,
                                              action_space))

    def stop_forward(self, observation):
        ''' return Q(s, a) where a is the stop action
        '''
        return self.stop_predictor.forward(
            self.state_action_feature.stop_forward(observation))

    def batch_forward(self, obs_action_list, action_space):
        return self.predictor.batch_forward(self.state_action_feature.batch_forward(
            obs_action_list,
            action_space)).reshape(-1)

    def reset_parameters(self, **kwargs):
        self.predictor.reset_parameters()


class MLPActionValueFunction(TwoStageActionValueFunction):

    def __init__(self,
                 state_action_feature,
                 seed,
                 mlp_kwargs,
                 device='cpu',
                 **kwargs):
        if 'in_dim' in mlp_kwargs:
            mlp_kwargs.pop('in_dim')
        predictor = MultiLayerPerceptron(in_dim=state_action_feature.out_dim,
                                         seed=seed,
                                         device=device,
                                         **mlp_kwargs)
        super().__init__(predictor=predictor,
                         state_action_feature=state_action_feature,
                         device=device)

    def delete_rng(self):
        self.predictor.delete_rng()


class LinearSparseActionValueFunction(TwoStageActionValueFunction):

    '''
    linear predictor

    Attributes
    ----------
    use_zdd : bool
        if True, construct zdd for fitted q iteration
    '''

    def __init__(self,
                 state_action_feature,
                 device='cpu'):
        predictor = LinearSparsePredictor(
            state_action_feature.out_dim,
            device=device)
        super().__init__(predictor, state_action_feature)

    def greedy_forward(self, observation, action_space, construct_zdd=False):
        ''' return (max_a Q(s, a), argmax_a Q(s, a))
        '''
        if construct_zdd:
            q_max = torch.tensor(-float('inf'), device=self.device)
            best_action = None
            feature_list = []
            file_path_obj = self.state_action_feature.fingerprint.get_zdd_path(
                observation['current_mol'])
            for each_action in action_space.legal_go_action_generator(observation):
                try:
                    feature = self.state_action_feature.forward(
                        observation,
                        each_action,
                        action_space)
                    no_torch_feature = feature.cpu().tolist()
                    q_val = self.predictor.forward(feature)
                    no_torch_feature = set(no_torch_feature)
                    feature_list.append(no_torch_feature)
                    if q_max.item() < q_val.item():
                        q_max = q_val
                        best_action = each_action
                except ValueError:
                    pass
            q_stop = self.stop_forward(observation)
            if q_max.item() < q_stop.item():
                q_max = q_stop
                best_action = action_space.stop_action

            if not file_path_obj.exists():
                from graphillion import setset
                my_setset = setset(feature_list)
                output = (my_setset.dumps(), my_setset.universe())
                with gzip.open(file_path_obj, 'wb') as f:
                    pickle.dump(output, f)
                del setset
                del my_setset
                del output
                import gc
                gc.collect()
            return q_max, best_action
        else:
            return super().greedy_forward(observation, action_space)

    def greedy_predict(self, observation, action_space):
        ''' return max_a Q(s, a)
        '''
        if observation is None:
            return torch.tensor([0.])
        if hasattr(self.state_action_feature.fingerprint, 'dir_obj'):
            key = self.state_action_feature.obs2key(observation)
            if self.state_action_feature.fingerprint.zdd_exists(key):
                _q_max = self.state_action_feature.light_greedy_predict(
                    observation,
                    self.state_action_feature.fingerprint.get_zdd_path(key),
                    self.predictor,
                    self.state_action_feature.fingerprint)
                return _q_max
        return self.greedy_forward(observation, action_space)[0]

    def fit(self,
            state_action_train_list,
            y_train,
            action_space,
            model_name,
            model_kwargs,
            kmm_kwargs,
            state_action_test_list=None,
            logger=print):
        X_sparse_list = [
            self.state_action_feature.forward(each_obs,
                                              each_action,
                                              action_space)
            for each_obs, each_action in state_action_train_list]
        if state_action_test_list:
            X_sparse_test_list = [
                self.state_action_feature.forward(each_obs,
                                                  each_action,
                                                  action_space)
                for each_obs, each_action in state_action_test_list]
        else:
            X_sparse_test_list = []
        return self.predictor.fit(
            X_sparse_list,
            y_train,
            model_name,
            model_kwargs,
            kmm_kwargs,
            X_sparse_test_list,
            logger)
