#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'


from abc import abstractmethod
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import math
import itertools
import numpy as np
import torch
from molgym.chem.qsar import SAScoreMixin
from molgym.utils import construct_dataset
from mlbasics.utils import RngMixin, OptimizerMixin
from ..env import simulate
from ..episode import Transition, Episode, EpisodeMemory


class AgentBase(RngMixin, OptimizerMixin, SAScoreMixin):

    ''' Base class for agents,
    implementing common methods

    Attributes
    ----------
    discount : float
        discounting factor of reward
    '''

    def __init__(self,
                 seed,
                 discount=1.0,
                 max_step=10,
                 device='cpu',
                 **kwargs):
        # do not use `action_space` as an instance var.
        # its data size could be huge.
        self.discount = discount
        self.max_step = max_step
        self.device = device
        self.kwargs = kwargs
        self.set_seed(seed)

    def to(self, device):
        self.device = device

    def act(self, observation, action_space):
        if observation['n_step'] >= self.max_step\
           or observation['terminal'] == 1:
            return action_space.stop_action
        return self._act(observation, action_space)

    def criticize(self, observation, action, action_space):
        ''' return an estimate of action value
        '''
        return 0

    @abstractmethod
    def _act(self, observation, action_space):
        raise NotImplementedError

    def fit(self, **kwargs):
        return self

    def gen_episode(self, env, current_mol=None, random=False, remove_last=False) -> Episode:
        ''' generate one episode in `env` according to this agent

        Parameters
        ----------
        env : gym.Env
        current_mol : Mol
            initial molecule

        Returns
        -------
        Episode
        '''
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
                    action_dict = self.act(obs, env.action_space)
                next_obs, reward, done, _ = env.step(action_dict)
                action_dict_cpu = env.action_space.send_action_to(action_dict, 'cpu')
                critic = float(self.criticize(obs, action_dict, env.action_space))
                episode.append(Transition(state=obs,
                                          action=action_dict_cpu,
                                          next_state=next_obs,
                                          reward=reward,
                                          done=done,
                                          info={'critic': critic}))
                obs = next_obs
        if episode.reward_list[-1] != 0 and remove_last:
            episode.pop()
        return episode

    def gen_episode_memory(self, env, n_episode, random=False, remove_last=False,
                           workers=1, shutdown_everytime=True, **kwargs) -> EpisodeMemory:
        ''' generate episode memory consisting of `n_episode`

        Parameters
        ----------
        env : gym.Env
        n_episode : int
            the number of episodes to generate

        Returns
        -------
        EpisodeMemory
        '''
        episode_list = []
        initial_mol_list = env.observation_space.sample_mol_list(n_episode)

        def _multiple_gen_episode(env, initial_mol_list, random, remove_last, **kwargs):
            episode_list = []
            for each_init_mol in initial_mol_list:
                episode_list.append(self.gen_episode(env,
                                                     each_init_mol,
                                                     random,
                                                     remove_last,
                                                     **kwargs))
            return episode_list

        if workers == 1:
            episode_list = _multiple_gen_episode(env,
                                                 initial_mol_list,
                                                 random=random,
                                                 remove_last=remove_last,
                                                 **kwargs)
        else:
            if shutdown_everytime:
                for each_batch in range(math.ceil(len(initial_mol_list) // workers)):
                    episode_list += Parallel(
                        n_jobs=workers,
                        verbose=10)([delayed(self.gen_episode)(
                            env,
                            current_mol=each_mol,
                            random=random,
                            remove_last=remove_last,
                            **kwargs)
                                     for each_mol in initial_mol_list[each_batch * workers
                                                                      : (each_batch + 1) * workers]])
                    get_reusable_executor().shutdown(wait=True)
            else:
                batch_size = math.ceil(len(initial_mol_list) / workers)
                episode_list = list(itertools.chain.from_iterable(Parallel(
                    n_jobs=workers,
                    verbose=10)([delayed(_multiple_gen_episode)(
                        env,
                        initial_mol_list[batch_size * each_worker_idx : batch_size * (each_worker_idx + 1)],
                        random=random,
                        remove_last=remove_last,
                        **kwargs)
                                 for each_worker_idx in range(workers)])))
        '''
        for each_idx in range(n_episode):
            episode_list.append(self.gen_episode(
                env,
                current_mol=initial_mol_list[each_idx],
                random=random,
                remove_last=remove_last,
                **kwargs))
        '''
        return EpisodeMemory(episode_list, seed=self.gen_seed())


class RandomAgent(AgentBase):

    ''' agent selecting random actions
    '''

    def _act(self, observation, action_space):
        return action_space.sample()


class LegalRandomAgent(AgentBase):

    ''' agent selecting random legal actions
    '''

    def _act(self, observation, action_space):
        ''' act randomly but legally

        Parameters
        ----------
        observation : tuple (Mol, int)
            observation[0] corresponds to the current molecule
            observation[1] corresponds to the number of reactions applied so far

        Returns
        -------
        action
        '''
        return action_space.legal_sample(observation)


class LegalGreedyAgent(AgentBase):

    ''' assuming the qsar model is available, this agent greedily chooses the next actions
    '''

    def __init__(self,
                 qsar_model,
                 seed,
                 discount=1.0,
                 max_step=10,
                 device='cpu',
                 **kwargs):
        super().__init__(seed=seed,
                         discount=discount,
                         max_step=max_step,
                         device=device)
        self.qsar_model = qsar_model

    def _act(self, observation, action_space):
        ''' act greedily

        Parameters
        ----------
        observation : tuple (Mol, int)
            observation[0] corresponds to the current molecule
            observation[1] corresponds to the number of reactions applied so far

        Returns
        -------
        action_dict
        '''
        current_mol = observation['current_mol'] #Chem.MolFromSmiles(observation)
        best_action = action_space.stop_action
        best_tgt = self.qsar_model.forward(current_mol)

        for each_action in action_space.legal_go_action_generator(current_mol):
            try:
                product_mol, success = simulate(current_mol,
                                                each_action,
                                                action_space,
                                                self.sascorer)
                if success:
                    target_val = self.qsar_model.forward(product_mol)
                    if target_val > best_tgt:
                        best_action = each_action
                        best_tgt = target_val
            except:
                pass
        return best_action

'''
greedy agent to increase similarity to the data set
'''

class LegalRandomAgentWithOfflineData(LegalRandomAgent):

    ''' LegalRandomAgent + ability to generate episode memory from offline data
    '''

    def __init__(self,
                 working_dir,
                 csv_path,
                 mol_col,
                 tgt_col,
                 preprocessing_list,
                 seed,
                 discount=1.0,
                 max_step=10,
                 rate=5.0,
                 offline_as_done=True,
                 device='cpu',
                 **kwargs):
        super().__init__(seed=seed,
                         discount=discount,
                         max_step=max_step,
                         device=device)
        self.mol_list, self.tgt_list \
            = construct_dataset(
                str(working_dir / csv_path),
                mol_col,
                tgt_col,
                preprocessing_list)
        self.rate = rate
        self.offline_as_done = offline_as_done
        self.idx = 0
        self.n_episode = len(self.mol_list)

    def gen_offline_episode_memory(self, env) -> EpisodeMemory:
        ''' generate EpisodeMemory from offline data.
        each data is regarded as a measurement state transition
        '''
        episode_list = []
        for each_mol, each_tgt in zip(self.mol_list, self.tgt_list):
            episode = Episode()
            done = self.offline_as_done
            if np.isinf(self.max_step):
                n_step = self.rng.poisson(lam=self.rate)
                transition = Transition({'current_mol': each_mol,
                                         'n_step': n_step},
                                        env.action_space.stop_action,
                                        {'current_mol': each_mol,
                                         'n_step': n_step+1},
                                        each_tgt,
                                        done,
                                        info={'critic': 0})
            else:
                transition = Transition({'current_mol': each_mol,
                                         'n_step': self.max_step},
                                        env.action_space.stop_action,
                                        {'current_mol': each_mol,
                                         'n_step': self.max_step+1},
                                        each_tgt,
                                        done,
                                        info={'critic': 0})
            episode.append(transition)
            episode_list.append(episode)
        return EpisodeMemory(episode_list, seed=self.gen_seed())

    def gen_episode_memory(self,
                           env,
                           n_episode,
                           remove_last=True,
                           include_offline_data=True,
                           **kwargs):
        ''' generate episode memory consisting of
        `n_episode` random behavior and offline data

        Parameters
        ----------
        env : gym.Env
        n_episode : int
            the number of episodes to generate

        Returns
        -------
        EpisodeMemory
        '''
        '''
        episode_list = []
        for _ in range(n_episode):
            episode = self.gen_episode(env)
            # remove non-zero reward so as not to use the simulated reward
            if episode.reward_list[-1] != 0 and remove_last:
                episode.pop()
            episode_list.append(episode)
        '''
        '''
        offline_mol_list = self.rng.choice(self.mol_list,
                                           size=n_offline_episode,
                                           replace=True)
        for each_start_mol in offline_mol_list:
            episode = self.gen_episode(env, each_start_mol)
            # remove non-zero reward so as not to use the simulated reward
            if episode.reward_list[-1] != 0 and remove_last:
                episode.pop()
            episode_list.append(episode)
        random_memory = EpisodeMemory(episode_list, seed=self.gen_seed())
        '''
        random_memory = super().gen_episode_memory(env,
                                                   n_episode,
                                                   random=False,
                                                   remove_last=remove_last,
                                                   **kwargs)
        offline_memory = self.gen_offline_episode_memory(env)
        if include_offline_data:
            return random_memory + offline_memory
        return random_memory
