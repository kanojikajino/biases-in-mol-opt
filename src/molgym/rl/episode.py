#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from collections import namedtuple
from rdkit import Chem
import numpy as np
from mlbasics.utils import RngMixin
from ..utils import construct_dataset


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'info'))

class Episode:

    ''' data class of one episode

    Attributes
    ----------
    transition_list : list of `Transition` objects
    reward_list : list of floats
    '''

    def __init__(self):
        self.transition_list = []
        self.reward_list = []

    def append(self, transition: Transition):
        ''' append a new transition
        '''
        self.transition_list.append(transition)
        self.reward_list.append(transition.reward)

    def pop(self):
        transition = self.transition_list.pop()
        reward = self.reward_list.pop()
        return transition, reward

    def value(self, discount):
        ''' compute the value of this episode using `discount`

        Parameters
        ----------
        discount : float
            discounting factor for cumulative reward

        Returns
        -------
        float
            Value of the episode
        '''
        if discount > 1.0 or discount < 0.0:
            raise ValueError('discount must be in [0, 1]')
        reward_array = np.array(self.reward_list)
        val = 0
        for each_reward in reward_array[::-1]:
            val = discount * val + each_reward
        return val

    def __str__(self):
        txt = '{}\t{}\n'.format('iter', 'smiles')
        for each_idx, each_transition in enumerate(self.transition_list):
            txt += '{}\t{}\n'.format(each_idx,
                                     Chem.MolToSmiles(each_transition.state['current_mol']))
            txt += ' * action: {}\n'\
                .format(each_transition.action)
            txt += ' * env is done: {}\n'.format(each_transition.done)
            txt += ' * reward: {}\n'.format(self.reward_list[each_idx])
        return txt

    def __len__(self):
        return len(self.transition_list)


class EpisodeMemory(RngMixin):

    def __init__(self, episode_list, seed):
        self.set_seed(seed)
        self.episode_list = episode_list
        self.transition_list = []
        for each_episode in episode_list:
            self.transition_list += each_episode.transition_list
        self.n_transition = len(self.transition_list)

    def sample_minibatch(self, batch_size, n_step=None):
        if batch_size == -1:
            if n_step is None:
                return self.transition_list
            else:
                return self.get_transition_list(n_step)
        else:
            if n_step is None:
                idx_array = self.rng.choice(np.arange(self.n_transition),
                                            size=batch_size,
                                            replace=False)
                return [self.transition_list[each_idx] for each_idx in idx_array]
            else:
                n_transition_list = self.get_transition_list(n_step)
                idx_array = self.rng.choice(np.arange(len(n_transition_list)),
                                            size=batch_size,
                                            replace=False)
                return [n_transition_list[each_idx] for each_idx in idx_array]

    def __add__(self, other):
        self.episode_list = self.episode_list + other.episode_list
        self.transition_list = self.transition_list + other.transition_list
        self.n_transition = self.n_transition + other.n_transition
        return self

    def append(self, other):
        self.episode_list.extend(other.episode_list)
        self.transition_list.extend(other.transition_list)
        self.n_transition += other.n_transition
        return self

    @property
    def done_transition_list(self):
        return [each_transition
                for each_transition in self.transition_list
                if each_transition.done]

    @property
    def done_mol_list(self):
        return [each_transition.state['current_mol'] for each_transition in self.done_transition_list]

    @property
    def undone_transition_list(self):
        return [each_transition
                for each_transition in self.transition_list
                if not each_transition.done]

    def get_transition_list(self, n_step):
        return [each_transition for each_transition in self.transition_list
                if each_transition.state['n_step'] == n_step]

    def get_mol_list(self, n_step):
        return [each_transition.state['current_mol'] for each_transition in self.get_transition_list(n_step)]


    def resample_episode(self):
        resampled_episode_list = self.rng.choice(self.episode_list,
                                                 size=len(self.episode_list),
                                                 replace=True).tolist()
        return EpisodeMemory(resampled_episode_list, seed=self.gen_seed())

    def train_test_random_split(self, test_ratio):
        if test_ratio <= 0 or test_ratio >= 1:
            raise ValueError('test_ratio must be in (0, 1).')
        train_ratio = 1. - test_ratio
        n_total = len(self.episode_list)
        n_train = int(n_total * train_ratio)
        n_test = n_total - n_train
        if n_train == 0 or n_test == 0:
            raise ValueError('n_train or n_test become 0. please fix test_ratio accordingly.')
        perm_episode_list = self.rng.permutation(self.episode_list)
        return (EpisodeMemory(perm_episode_list[:n_train], seed=self.gen_seed()),
                EpisodeMemory(perm_episode_list[n_train:], seed=self.gen_seed()))
    
    def extract_dataset(self):
        mol_list = []
        tgt_list = []
        for each_transition in self.transition_list:
            if each_transition.done:
                mol_list.append(each_transition.state['current_mol'])
                tgt_list.append(each_transition.reward)
        return mol_list, tgt_list



def episode_memory_from_offline_data(env, seed, working_dir, csv_path, mol_col, tgt_col, preprocessing_list):
    mol_list, tgt_list \
            = construct_dataset(
                str(working_dir / csv_path),
                mol_col,
                tgt_col,
                preprocessing_list)
    episode_list = []
    for each_mol, each_tgt in zip(mol_list, tgt_list):
        episode = Episode()
        done = True
        transition = Transition({'current_mol': each_mol,
                                 'n_step': env.max_step},
                                env.action_space.stop_action,
                                {'current_mol': each_mol,
                                 'n_step': env.max_step+1},
                                each_tgt,
                                done,
                                info={'critic': 0})
        episode.append(transition)
        episode_list.append(episode)
    return EpisodeMemory(episode_list, seed=seed)
