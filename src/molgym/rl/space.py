#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from itertools import product
from gym.spaces import Discrete, MultiDiscrete, Dict, Space, Box, Tuple
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
import torch
from ..chem.reaction import ReactionCorpus
from ..chem.qsar import SAScoreMixin
from mlbasics.utils import RngMixin

# supress errors
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


class OneHotSpace(Discrete):

    def idx2array(self, idx):
        array = torch.zeros(self.n)
        array[idx] = 1.0
        return array

    @staticmethod
    def array2idx(array):
        return torch.where(array == 1.0)[0].item()

    def sample(self):
        return self.idx2array(super().sample())

    def contains(self, x):
        return super().contains(self.array2idx(x))


class MoleculeSpace(Dict, RngMixin):

    def __init__(self, reactant_path, max_step, seed):
        self.set_seed(seed)
        self.max_step = max_step
        self.reactant_list = [each_mol for each_mol \
                              in Chem.SmilesMolSupplier(reactant_path, titleLine=False)]
        super().__init__({'current_mol': Space(seed=self.gen_seed()),
                          'n_step': Discrete(max_step+1, seed=self.gen_seed()),
                          'terminal': Discrete(2, seed=self.gen_seed())})

    def contains(self, observation):
        return isinstance(observation['current_mol'], rdchem.Mol) \
            and 0 <= observation['n_step'] <= self.max_step\
            and 0 <= observation['terminal'] <= 1

    def sample(self):
        return {'current_mol': self.rng.choice(self.reactant_list),
                'n_step': self.rng.integers(0, self.max_step),
                'terminal': self.rng.integers(0, 1)}

    def sample_mol_list(self, size):
        return self.rng.choice(self.reactant_list, size=size, replace=True)


class ReactionSpace(Dict, RngMixin, SAScoreMixin):

    ''' Corpus of reaction objects and reactants

    Attributes
    ----------
    reaction_tsv_path : str
        path to reaction templates in tsv format.
        each row contains the name of a reaction and its SMARTS representation
    reactant_path
    '''

    stop_action = {'stop': 1, 'reaction_idx': None, 'current_mol_pos': None, 'reactant_list': None}

    def __init__(self, reactant_path, reaction_tsv_path, seed, max_reactant=float('inf'),
                 allow_multiple_matches=False):
        self.set_seed(seed)
        self.reaction_corpus = ReactionCorpus(reactant_path,
                                              reaction_tsv_path,
                                              max_reactant,
                                              allow_multiple_matches)
        super().__init__(
            {'stop': Discrete(2, seed=self.gen_seed()),
             'reaction_idx': Discrete(self.reaction_corpus.n_reaction, seed=self.gen_seed()),
             'current_mol_pos': Discrete(self.reaction_corpus.max_reactant, seed=self.gen_seed()),
             'reactant_list': MultiDiscrete([self.reaction_corpus.n_reactant] \
                                            * (self.reaction_corpus.max_reactant-1), seed=self.gen_seed())})

    def get_reactant_list(self, n_reactant, replace=False):
        return self.rng.choice(self.reaction_corpus.reactant_list, n_reactant, replace=replace).tolist()

    def sample(self):
        ''' randomly sampling actions, irrespective of legality

        Returns
        -------
        action : dict
        '''
        go_nogo = self.rng.choice([0, 1])
        reaction_idx = self.rng.integers(self.reaction_corpus.n_reaction)
        current_mol_pos = self.rng.integers(
            self.reaction_corpus.reaction_list[reaction_idx].n_reactant)
        reactant_list = []
        for _ in range(self.reaction_corpus.max_reactant-1):
            reactant_list.append(self.rng.integers(self.reaction_corpus.n_reactant))
        return {'stop': go_nogo,
                'reaction_idx': reaction_idx,
                'current_mol_pos': current_mol_pos,
                'reactant_list': reactant_list}

    def legal_sample(self, observation, device='cpu'):
        ''' Given the current molecule, randomly sampling legal action.
        If there exists no legal action, it sets `stop=1` and determines other actions randomly.
        In this case, the environment applies the random action and then stops.

        Parameters
        ----------
        current_mol : Mol

        Returns
        -------
        action : dict
        '''
        current_mol = observation['current_mol']
        legal_reaction_list = self.reaction_corpus.legal_reaction(current_mol)
        if not legal_reaction_list:
            # if there is no legal move, stop the generation.
            # actions other than `stop` can be set arbitrarily;
            # they are ignored in the environment.
            success = False
        else:
            max_trial = min(5, len(legal_reaction_list))
            for _ in range(max_trial):
                #go_nogo = self.rng.choice([0, 1])
                go_nogo = 0
                reaction_dict = self.rng.choice(legal_reaction_list)
                reaction_idx = reaction_dict['reaction_idx']
                current_mol_pos = self.rng.choice(reaction_dict['current_mol_pos'])
                reactant_list = []
                for each_position in range(len(reaction_dict['candidate_reactant'])):
                    if each_position != current_mol_pos:
                        reactant_idx = self.rng.choice(
                            reaction_dict['candidate_reactant'][each_position])
                        reactant_list.append(reactant_idx)
                    else:
                        pass
                action_dict = {'stop': go_nogo,
                               'reaction_idx': reaction_idx,
                               'current_mol_pos': current_mol_pos,
                               'reactant_list': reactant_list}
                from .env import simulate
                _, success = simulate(current_mol, action_dict, self, self.sascorer)
                if success:
                    break
                #if not success:
                #    raise ValueError('simulation fails')
        if not success:
            action_dict = self.stop_action
        return action_dict

    def legal_go_action_generator(self, observation):
        current_mol = observation['current_mol']
        legal_reaction_list = self.reaction_corpus.legal_reaction(current_mol)
        if legal_reaction_list:
            stop = 0
            for each_reaction_dict in legal_reaction_list:
                reaction_idx = each_reaction_dict['reaction_idx']
                for each_current_mol_pos in each_reaction_dict['current_mol_pos']:
                    reactant_list_list = []
                    for each_position in range(len(each_reaction_dict['candidate_reactant'])):
                        if each_position != each_current_mol_pos:
                            reactant_list_list.append(
                                each_reaction_dict['candidate_reactant'][each_position])
                        else:
                            pass
                    for each_reactant_list in product(*reactant_list_list):
                        action_dict = {
                            'stop': stop,
                            'reaction_idx': reaction_idx,
                            'current_mol_pos': each_current_mol_pos,
                            'reactant_list': list(each_reactant_list)}
#                            + [0] * (self.max_reactant - len(each_reaction_dict['candidate_reactant']))}
                        yield action_dict

    @staticmethod
    def send_action_to(action, device):
        return action

    def to(self, device):
        pass


class ContinuousReactionSpace(Tuple, RngMixin):

    stop_action = (None, None)

    def __init__(self,
                 low,
                 high,
                 action_dim,
                 reactant_path,
                 reaction_tsv_path,
                 seed,
                 max_reactant=2,
                 allow_multiple_matches=False,
                 device='cpu'):
        self.set_seed(seed)
        self.device = device
        self.reaction_corpus = ReactionCorpus(reactant_path=reactant_path,
                                              reaction_tsv_path=reaction_tsv_path,
                                              max_reactant=max_reactant,
                                              allow_multiple_matches=allow_multiple_matches)
        template_action_space = OneHotSpace(self.reaction_corpus.n_reaction,
                                            seed=self.gen_seed())
        reactant_action_space = Box(low, high, shape=(action_dim,), seed=self.gen_seed())
        #reactant_action_space.shape = reactant_action_space._shape
        self.out_dim = self.reaction_corpus.n_reaction + action_dim
        super().__init__([template_action_space,
                          reactant_action_space])

    def get_reactant_list(self, n_reactant, replace=False):
        return self.rng.choice(self.reaction_corpus.reactant_list, n_reactant, replace=replace).tolist()

    def legal_sample(self, observation):
        ''' Given the current molecule, randomly sampling legal action.

        Parameters
        ----------
        current_mol : Mol

        Returns
        -------
        action : dict
        '''
        current_mol = observation['current_mol']
        try:
            legal_reaction_idx_list = list(set(list(zip(
                *self.reaction_corpus.legal_reaction_template(current_mol)))[0]))
        except IndexError:
            return self.stop_action
        reaction_idx = self.rng.choice(legal_reaction_idx_list)
        reaction_array = torch.zeros(self.reaction_corpus.n_reaction, device=self.device)
        reaction_array[reaction_idx] = 1.0
        return reaction_array, torch.tensor(self.spaces[1].sample(), device=self.device)

    def send_action_to(self, action, device):
        if action == self.stop_action:
            return action
        return action[0].to(device), action[1].to(device)

    def to(self, device):
        self.device = device


class DiscreteReactionSpace(Tuple, RngMixin, SAScoreMixin):

    stop_action = (None, None)

    def __init__(self,
                 reactant_path,
                 reaction_tsv_path,
                 seed,
                 max_reactant=2,
                 allow_multiple_matches=False,
                 device='cpu'):
        self.set_seed(seed)
        self.device = device
        self.reaction_corpus = ReactionCorpus(reactant_path=reactant_path,
                                              reaction_tsv_path=reaction_tsv_path,
                                              max_reactant=max_reactant,
                                              allow_multiple_matches=allow_multiple_matches)
        template_action_space = Discrete(self.reaction_corpus.n_reaction,
                                         seed=self.gen_seed())
        reactant_action_space = Discrete(self.reaction_corpus.n_reactant,
                                         seed=self.gen_seed())
        #reactant_action_space.shape = reactant_action_space._shape
        self.out_dim = self.reaction_corpus.n_reaction + self.reaction_corpus.n_reactant
        super().__init__([template_action_space,
                          reactant_action_space])

    def get_reactant_list(self, n_reactant, replace=False):
        return self.rng.choice(self.reaction_corpus.reactant_list, n_reactant, replace=replace).tolist()

    def legal_sample(self, observation):
        ''' Given the current molecule, randomly sampling legal action.

        Parameters
        ----------
        current_mol : Mol

        Returns
        -------
        action : dict
        '''
        current_mol = observation['current_mol']
        legal_reaction_list = self.reaction_corpus.legal_reaction(current_mol)
        if not legal_reaction_list:
            # if there is no legal move, stop the generation.
            # actions other than `stop` can be set arbitrarily;
            # they are ignored in the environment.
            success = False
        else:
            max_trial = min(5, len(legal_reaction_list))
            for _ in range(max_trial):
                #go_nogo = self.rng.choice([0, 1])
                reactant_idx = self.rng.integers(self.reaction_corpus.n_reactant)
                reaction_dict = self.rng.choice(legal_reaction_list)
                reaction_idx = reaction_dict['reaction_idx']
                current_mol_pos = self.rng.choice(reaction_dict['current_mol_pos'])
                reactant_list = []
                for each_position in range(len(reaction_dict['candidate_reactant'])):
                    if each_position != current_mol_pos:
                        reactant_idx = self.rng.choice(
                            reaction_dict['candidate_reactant'][each_position])
                        reactant_list.append(reactant_idx)
                    else:
                        pass
                action_dict = {'stop': 0,
                               'reaction_idx': reaction_idx,
                               'current_mol_pos': current_mol_pos,
                               'reactant_list': reactant_list}
                from .env import simulate
                _, success = simulate(current_mol, action_dict, self, self.sascorer)
                if success:
                    #reaction_array = torch.zeros(self.reaction_corpus.n_reaction, device=self.device)
                    #reaction_array[reaction_idx] = 1.0
                    #reactant_array = torch.zeros(self.reaction_corpus.n_reactant, device=self.device)
                    #try:
                    #    reactant_array[reactant_idx] = 1.0
                    #except:
                    #    pass
                    #action = (reaction_array, reactant_array)
                    action = (reaction_idx, reactant_idx)
                    break
                #if not success:
                #    raise ValueError('simulation fails')
        if not success:
            action = self.stop_action
        return action

    def send_action_to(self, action, device):
        ''' action is a list of int
        '''
        return action

    def to(self, device):
        self.device = device

"""
class GrammaticalEvolutionStateSpace:
    pass

class GrammaticalEvolutionActionSpace(Discrete, RngMixin):

    stop_action = (None, None)

    def __init__(self,
                 hrg,
                 seed,
                 device='cpu'):
        self.set_seed(seed)
        self.device = device
        self.hrg = hrg
        super().__init__(self.hrg.num_prod_rule, seed=self.seed)

    def legal_sample(self, observation):
        ''' Given the current molecule, randomly sampling legal action.

        Parameters
        ----------
        current_mol : Mol

        Returns
        -------
        action : dict
        '''
        current_mol = observation['current_mol']
        try:
            legal_reaction_idx_list = list(set(list(zip(
                *self.reaction_corpus.legal_reaction_template(current_mol)))[0]))
        except IndexError:
            return self.stop_action
        reaction_idx = self.rng.choice(legal_reaction_idx_list)
        reaction_array = torch.zeros(self.reaction_corpus.n_reaction, device=self.device)
        reaction_array[reaction_idx] = 1.0
        return reaction_array, torch.tensor(self.spaces[1].sample(), device=self.device)

    def send_action_to(self, action, device):
        if action == self.stop_action:
            return action
        return action[0].to(device), action[1].to(device)

    def to(self, device):
        self.device = device
"""
