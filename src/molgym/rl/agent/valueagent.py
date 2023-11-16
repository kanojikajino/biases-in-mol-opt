#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
value-based agent
'''

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from copy import deepcopy
import time
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import numpy as np
from rdkit import Chem
import torch
from .base import AgentBase
from ..episode import Episode, Transition
from ..valfunc import ZeroActionValueFunction


class FittedQIterationAgentBase(AgentBase):

    ''' Base class for fitted Q iteration agent
    '''

    def __init__(self,
                 action_val_model,
                 seed,
                 discount=1.0,
                 max_step=10,
                 **kwargs):
        super().__init__(seed=seed,
                         discount=discount,
                         max_step=max_step)
        self.init_action_val(action_val_model)

    def init_action_val(self, action_val_model):
        ''' initialize action value functions according to `max_step`
        '''
        if np.isinf(self.max_step):
            self._action_val = action_val_model
            self._target_action_val = deepcopy(action_val_model)
            self._target_action_val.eval()
        else:
            self.action_val_list = []
            for _ in range(self.max_step+1):
                self.action_val_list.append(deepcopy(action_val_model))
            # for target
            self.action_val_list.append(ZeroActionValueFunction())

    def action_val(self, n_step):
        ''' this evaluates the value of each action when the state is at `n_step`.

        Parameters
        ----------
        n_step : int
            the number of chemical reactions applied so far

        Returns
        -------
        action value function
        '''
        if np.isinf(self.max_step):
            return self._action_val
        return self.action_val_list[n_step]

    def target_val(self, n_step):
        ''' this evaluates the value of each action when the state is at `n_step`.

        Parameters
        ----------
        n_step : int
            the number of chemical reactions applied so far

        Returns
        -------
        action value function
        '''
        if np.isinf(self.max_step):
            return self._target_action_val
        return self.action_val_list[n_step+1]

    def update_target(self):
        if np.isinf(self.max_step):
            self._target_action_val.load_state_dict(
                self._action_val.state_dict())

    def loss(self, transition, n_step):
        ''' compute loss between the n-th and (n+1)-th action value functions

        Parameters
        ----------
        transition : Transition
        n_step : int
            this loss is used to train the action value function used when the state is at `n_step`.
        '''
        if n_step == self.max_step == transition.state['n_step']\
           or (n_step < self.max_step and transition.state['n_step'] < self.max_step):
            diff = self.action_val(n_step).forward(transition.state,
                                                   transition.action)
            with torch.no_grad():
                greedy = self.target_val(n_step).greedy_predict(
                    transition.next_state)
            diff = diff - (transition.reward
                           + self.discount * greedy)
            return 0.5 * (diff ** 2)
        raise ValueError('trainsition and n_step are not compatible')

    def _act(self, observation, action_space):
        return self.action_val(observation['n_step']).greedy_action(observation,
                                                                    action_space)

    def criticize(self, observation, action, action_space):
        return self.action_val(observation['n_step']).forward(observation,
                                                              action,
                                                              action_space)


class StochasticFittedQIterationAgent(FittedQIterationAgentBase):

    ''' Optimization by stochastic gradient descent
    '''

    def fit(self,
            episode_memory,
            n_epochs=100,
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 1e-2},
            tgt_update_freq=50,
            print_freq=1,
            logger=print):
        if np.isfinite(self.max_step):
            n_epochs = self.max_step
        optimizer = getattr(torch.optim, optimizer)(
            params=self.action_val.parameters(),
            **optimizer_kwargs)

        running_loss = 0
        n_examples = 0
        for n_step in range(n_epochs, -1, -1): # n_epochs, n_epochs-1,...,0
            for each_transition in episode_memory.transition_list:
                try:
                    loss = self.loss(each_transition, n_step)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    n_examples += 1
                except ValueError:
                    pass
            if (n_epochs - n_step + 1) % print_freq == 0:
                logger('#(step) = {}\t loss = {}'.format(
                    n_step,
                    running_loss/n_examples))
                running_loss = 0
                n_examples = 0
            if (n_epochs - n_step + 1) % tgt_update_freq == 0:
                self.update_target()
                logger(' ** update target network **')
        #del self.target_action_val
        return self


class ExactFittedQIterationAgent(FittedQIterationAgentBase):

    def init_action_val(self, action_val_model):
        ''' initialize action value functions according to `max_step`
        '''
        if np.isinf(self.max_step):
            self._action_val = action_val_model
        else:
            self.action_val_list = []
            for _ in range(self.max_step+1):
                self.action_val_list.append(deepcopy(action_val_model))
            # for target
            self.action_val_list.append(ZeroActionValueFunction())
        self._target_action_val = deepcopy(action_val_model)
        self._target_action_val.eval()

    @property
    def setset_action_val(self):
        return self._target_action_val

    def fit(self,
            episode_memory,
            env,
            n_behavior_updates=10,
            n_epochs=100,
            incremental_episode_size=100,
            model_name='RidgeCV',
            model_kwargs={'alphas': [1e0, 1e+1, 1e+2]},
            kmm_kwargs={'kernel_name': 'TanimotoSimilarity',
                        'kernel_kwargs': {},
                        'max_beta': 1e+1,
                        'tol': 1e-3},
            covariate_shift=True,
            print_freq=1,
            construct_dataset_workers=1,
            gen_episode_workers=1,
            remove_last=True,
            skip_second_update=False,
            logger=print,
            **kwargs):
        if np.isfinite(self.max_step):
            n_epochs = self.max_step
        action_space = env.action_space
        running_loss = 0
        n_iters = 0
        tot_time = 0
        for iter_idx in range(n_behavior_updates):
            logger('\n##### behavior update: {} #####'.format(iter_idx))
            for n_step in range(n_epochs, -1, -1): # n_epochs, n_epochs-1,...,0
                if n_step == n_epochs - 1 and skip_second_update:
                    logger(' * target action value function is copied into the action value function to be trained.')
                    self.action_val(n_step).load_state_dict(self.target_val(n_step).state_dict())
                else:
                    t_start = time.time()
                    state_action_train_list, y_train, state_action_test_list \
                        = self.construct_dataset(
                            episode_memory,
                            action_space,
                            n_step,
                            construct_dataset_workers,
                            covariate_shift=covariate_shift)
                    logger(' * #(training examples) = {}'.format(len(state_action_train_list)))
                    loss = self.action_val(n_step).fit(
                        state_action_train_list=state_action_train_list,
                        y_train=y_train,
                        action_space=action_space,
                        state_action_test_list=state_action_test_list,
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        kmm_kwargs=kmm_kwargs,
                        logger=logger)
                    running_loss += loss.item()
                    n_iters += 1
                    if (n_epochs - n_step + 1) % print_freq == 0:
                        logger('#(step) = {}\t loss = {}'.format(
                            n_step,
                            running_loss/n_iters))
                        running_loss = 0
                        n_iters = 0
                    self.update_target()
                    logger(' ** update target network **')
                    t_end = time.time()
                    logger(' * n_step: {}\t{} sec'.format(n_step, t_end-t_start))
                    tot_time += (t_end - t_start)
                if n_step == n_epochs:
                    # set stop_predictor
                    for _n_step in range(n_epochs, -1, -1):
                        self.action_val(_n_step).stop_predictor \
                            = self.action_val(n_epochs).predictor
            if iter_idx != n_behavior_updates - 1:
                additional_episode_memory = self.gen_episode_memory(
                    env,
                    incremental_episode_size,
                    remove_last=remove_last,
                    construct_zdd=self.action_val(0).state_action_feature.fingerprint.use_zdd,
                    workers=gen_episode_workers)
                episode_memory = episode_memory + additional_episode_memory
            get_reusable_executor().shutdown(wait=True)

        logger(' * total time: {} sec'.format(tot_time))
        return self

    def gen_episode(self, env, current_mol=None, random=False, remove_last=False, construct_zdd=False):
        if not construct_zdd:
            return super().gen_episode(env, current_mol, random, remove_last)
        obs = env.reset(observation={'current_mol': current_mol,
                                     'n_step': 0,
                                     'terminal': 0})
        episode = Episode()
        reward = 0
        done = False
        with torch.no_grad():
            while not done:
                if random:
                    action_dict = env.action_space.legal_sample(obs)
                else:
                    action_dict = self.act(
                        obs,
                        env.action_space,
                        construct_zdd=construct_zdd if obs['n_step'] != 0 else False)
                next_obs, reward, done, _ = env.step(action_dict)
                critic = float(self.criticize(obs, action_dict, env.action_space))
                episode.append(Transition(obs,
                                          env.action_space.send_action_to(action_dict, 'cpu'),
                                          next_obs,
                                          reward,
                                          done,
                                          {'critic': critic}))
                obs = next_obs
        if episode.reward_list[-1] != 0 and remove_last:
            episode.pop()
        return episode

    def act(self, observation, action_space, construct_zdd=False):
        if observation['n_step'] >= self.max_step:
            return action_space.stop_action
        return self._act(observation, action_space, construct_zdd)

    def _act(self, observation, action_space, construct_zdd=False):
        return self.action_val(observation['n_step']).greedy_forward(observation,
                                                                     action_space,
                                                                     construct_zdd)[1]

    def construct_dataset(self,
                          episode_memory,
                          action_space,
                          n_step,
                          workers,
                          covariate_shift=True):
        def _mol2smiles(mol):
            if mol is None:
                return None
            else:
                return Chem.MolToSmiles(mol)
        state_action_train_list = []
        y_train_list = []
        state_action_test_list = []

        test_transition_list = []
        # construct transition_list and greedy_list
        if n_step == self.max_step:
            transition_list = episode_memory.done_transition_list
            greedy_list = [torch.tensor([0.]) for each_transition in transition_list]
            test_transition_list = episode_memory.undone_transition_list
        elif n_step == self.max_step - 1:
            # this case, action is limited to `stop`
            transition_list = episode_memory.undone_transition_list
            greedy_list = [self.target_val(n_step).stop_forward(each_transition.next_state)
                           for each_transition in transition_list]
        else:
            if np.isinf(self.max_step):
                transition_list = episode_memory.transition_list
            else:
                transition_list = episode_memory.get_transition_list(n_step)
            if not self.setset_action_val.state_action_feature.fingerprint.dir_obj:
                if workers == 1:
                    greedy_list = [self.target_val(n_step).greedy_predict(
                        each_transition.next_state,
                        action_space) for each_transition in transition_list]
                else:
                    greedy_list = Parallel(n_jobs=workers, verbose=10)([delayed(self.target_val(n_step).greedy_predict)(
                        each_transition.next_state,
                        action_space) for each_transition in transition_list])
            else:
                if workers == 1:
                    greedy_list = [self.target_val(n_step).state_action_feature.light_greedy_predict(
                        each_transition.next_state,
                        self.setset_action_val.state_action_feature.fingerprint.get_zdd_path(
                            each_transition.next_state['current_mol']),
                        self.target_val(n_step).predictor,
                        self.target_val(n_step).state_action_feature.fingerprint)
                                   for each_transition in transition_list]
                else:
                    greedy_list = Parallel(n_jobs=workers, verbose=10)([
                        delayed(self.target_val(n_step).state_action_feature.light_greedy_predict)
                        (each_transition.next_state,
                         self.setset_action_val.state_action_feature.fingerprint.get_zdd_path(
                             each_transition.next_state['current_mol']),
                         self.target_val(n_step).predictor,
                         self.target_val(n_step).state_action_feature.fingerprint)
                        for each_transition in transition_list])
        get_reusable_executor().shutdown(wait=True)

        # construct data set
        for each_idx, each_transition in enumerate(transition_list):
            if greedy_list[each_idx] is not None:
                state_action_train_list.append((each_transition.state,
                                                each_transition.action))
                greedy = greedy_list[each_idx]
                try:
                    y_train_list.append((each_transition.reward
                                         + self.discount * greedy).item())
                except:
                    import pdb; pdb.set_trace()

        if covariate_shift:
            for each_transition in test_transition_list:
                state_action_test_list.append((each_transition.state,
                                               each_transition.action))
        return state_action_train_list, torch.tensor(y_train_list), state_action_test_list
