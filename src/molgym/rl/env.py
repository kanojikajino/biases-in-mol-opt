#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from abc import abstractmethod
from io import StringIO
import sys
import numpy as np
from gym import Env
from rdkit import Chem
import torch
import molgym
from .space import MoleculeSpace, ReactionSpace, ContinuousReactionSpace, DiscreteReactionSpace
from ..chem.qsar import SAScoreMixin
from mlbasics.utils import RngMixin


def simulate(mol,
             action,
             action_space,
             sascorer):
    ''' simulate state transition by applying `action` on `mol`

    Parameters
    ----------
    mol : Mol
    action : dict
    action_space : ReactionSpace
    sascorer : func, Mol -> float
        the smaller, the more synthetically accessible

    Returns
    -------
    Mol, bool
    '''
    success = True
    if action['stop'] == 1:
        # if done, no further reaction will be applied.
        product_mol = mol
    else:
        reaction = action_space.reaction_corpus.reaction_list[action['reaction_idx']]
        reactant_mol_list = []
        for each_reactant_idx in action['reactant_list']:
            reactant_mol_list.append(action_space.reaction_corpus.reactant_list[each_reactant_idx])
        reactant_mol_list.insert(action['current_mol_pos'], mol)
        assert reaction.n_reactant == len(reactant_mol_list)

        try:
            # each element in `product_mol_candidates` is a list of products
            #product_mol_candidates = reaction.apply_to([self.current_mol, reactant_mol])
            product_mol_candidates = reaction.apply_to(reactant_mol_list)
            if product_mol_candidates:
                if len(product_mol_candidates[0]) != 1:
                    raise NotImplementedError
                if len(product_mol_candidates) == 1:
                    product_mol = product_mol_candidates[0][0]
                else:
                    product_mol_list = []
                    for each_tuple in product_mol_candidates:
                        product_mol_list += list(each_tuple)
                    sascore_array = np.array(
                        sascorer.batch_forward(product_mol_list))
                    product_mol = product_mol_list[np.argmin(sascore_array)]
            else:
                raise ValueError
        except ValueError:
            # when reaction fails
            success = False
            product_mol = mol
    return product_mol, success



class ReactionEnvBase(Env, RngMixin, SAScoreMixin):

    ''' base class for a molecular reaction environment

    state = observation = current_mol, n_step

    Attributes
    ----------
    reactant_path : str
        path to a text file containing SMILES representations
        of building block molecules
    reaction_tsv_path : str
        path to a tsv file containing reaction templates in SMARTS format
    qsar_model : obj
        has function `forward`, which
        maps from Mol object to target property
    opt_orientation : `maximization` or `minimization`
        qsar output shold be maximized or minimized
    max_step : int
        maximum number of steps
    '''

    def __init__(self,
                 seed,
                 reactant_path=None,
                 reaction_tsv_path=None,
                 qsar_model=None,
                 opt_orientation='maximization',
                 max_step=10,
                 action_space=None,
                 allow_multiple_matches=False,
                 **kwargs):
        super().__init__()
        self.set_seed(seed)
        self.qsar_model = qsar_model
        self.opt_orientation = opt_orientation
        self.max_step = max_step
        self.observation_space = MoleculeSpace(reactant_path,
                                               max_step,
                                               seed=self.gen_seed())
        if action_space is None:
            self.action_space = ReactionSpace(reactant_path,
                                              reaction_tsv_path,
                                              seed=self.gen_seed(),
                                              allow_multiple_matches=allow_multiple_matches)
        else:
            self.action_space = action_space
        self.current_mol = None
        self.terminal = 0
        self.n_step = 0

    def to(self, device):
        return self

    def set_seed(self, seed):
        super().set_seed(seed)
        if hasattr(self, 'observation_space'):
            if hasattr(self.observation_space, 'set_seed'):
                self.observation_space.set_seed(self.gen_seed())
        if hasattr(self, 'action_space'):
            if hasattr(self.action_space, 'set_seed'):
                self.action_space.set_seed(self.gen_seed())

    @property
    def observation(self):
        return {'current_mol': self.current_mol,
                'n_step': self.n_step,
                'terminal': self.terminal}

    @property
    def is_maximization(self):
        return self.opt_orientation == 'maximization'

    @property
    def current_smiles(self):
        if hasattr(self, 'current_mol'):
            current_smiles = Chem.MolToSmiles(self.current_mol)
        else:
            current_smiles = None
        return current_smiles

    def reset(self, observation=None):
        if observation is None:
            observation = self.observation_space.sample()
            self.current_mol = observation['current_mol']
            self.n_step = 0
            self.terminal = 0
        else:
            if observation['current_mol'] is None:
                observation['current_mol'] = self.observation_space.sample()['current_mol']
            self.current_mol = observation['current_mol']
            self.n_step = observation['n_step']
            self.terminal = observation['terminal']
        return self.observation

    @abstractmethod
    def reward_model(self, observation, action, done, product_mol):
        pass

    def qsar2reward(self, mol):
        ''' helper function to convert QSAR score to reward,
        mainly caring about the orientation

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        float : reward, to be maximized
        '''
        if self.opt_orientation == 'maximization':
            reward = self.qsar_model.forward(mol)
        elif self.opt_orientation == 'minimization':
            reward = -self.qsar_model.forward(mol)
        else:
            raise ValueError('orientation of the optimization problem must be '\
                             'either maximization or minimization')
        return reward

    def simulate(self, observation, action, return_success=False):
        ''' simulate state transition by applying `action` on `state`,
        ignoring n_step
        '''
        done = (self.n_step >= self.max_step)
        terminal = int((action['stop'] == 1) or self.terminal == 1)
        if done or terminal:
            # if done, no further reaction will be applied.
            product_mol = self.current_mol
            success = True
        else:
            product_mol, success = simulate(observation['current_mol'],
                                            action,
                                            self.action_space,
                                            self.sascorer)
        if return_success:
            res = (product_mol, success)
        else:
            res = product_mol
        return res

    def step(self, action):
        ''' state transition and reward computation given an action.
        the environment accepts any action regardless of whether the action is legal or not.
        thus, it is the agent's responsibility to choose legal actions.

        Parameters
        ----------
        action : dict of go/no-go, reaction id, and building block id.

        Returns
        -------
        next_state : str
            SMILES
        reward : float
        done : bool
        info : dict
        '''
        done = (self.n_step >= self.max_step)
        self.terminal = int((action['stop'] == 1) or self.terminal == 1)
        if done or self.terminal:
            # if done, no further reaction will be applied.
            product_mol = self.current_mol
        else:
            product_mol = self.simulate(self.observation, action)

        # reward model must return penalty reward if current_mol = product_mol && done
        try:
            reward = self.reward_model(self.observation,
                                       action,
                                       done,
                                       product_mol)
        except ValueError:
            product_mol = self.current_mol
            reward = self.reward_model(self.observation,
                                       action,
                                       done,
                                       product_mol)
        self.current_mol = product_mol
        self.n_step += 1
        return self.observation, reward, done, {}

    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n {}'.format(self.current_smiles))
        return outfile


class ContinuousActionReactionEnv(ReactionEnvBase):

    def __init__(self,
                 n_lookup_reactants,
                 low,
                 high,
                 seed,
                 fingerprint_module='SelectedMolecularDescriptorsFingerprint',
                 fingerprint_kwargs={},
                 reactant_path=None,
                 reaction_tsv_path=None,
                 qsar_model=None,
                 opt_orientation='maximization',
                 max_step=10,
                 penalty_reward=-10.0,
                 allow_multiple_matches=False,
                 device='cpu',
                 **kwargs):
        self.device = device
        self.set_seed(seed)
        fingerprint = getattr(molgym.chem.fingerprint,
                              fingerprint_module)(**fingerprint_kwargs)
        action_dim = fingerprint.out_dim
        action_space = ContinuousReactionSpace(low, high, action_dim,
                                               reactant_path, reaction_tsv_path,
                                               seed=self.gen_seed(),
                                               max_reactant=2,
                                               allow_multiple_matches=allow_multiple_matches,
                                               device=device)
        try:
            self.reactant_array = fingerprint.fit_transform(action_space.reaction_corpus.reactant_list)
        except:
            self.reactant_array = fingerprint.batch_forward(action_space.reaction_corpus.reactant_list)
        self.reactant2idx_dict = {}
        for each_idx, each_mol in enumerate(action_space.reaction_corpus.reactant_list):
            self.reactant2idx_dict[Chem.MolToSmiles(each_mol)] = each_idx
        super().__init__(seed=self.gen_seed(),
                         reactant_path=reactant_path,
                         reaction_tsv_path=reaction_tsv_path,
                         qsar_model=qsar_model,
                         opt_orientation=opt_orientation,
                         max_step=max_step,
                         action_space=action_space)
        if n_lookup_reactants != 1:
            raise ValueError('n_lookup_reactants > 1 is deprecated.')
        self.n_lookup_reactants = n_lookup_reactants
        self.fingerprint = fingerprint
        self.penalty_reward = penalty_reward

    def to(self, device):
        self.device = device
        self.action_space.to(device)
        self.qsar_model.to(device)
        self.reactant_array = self.reactant_array.to(device)
        return self

    @property
    def reactant_list(self):
        return self.observation_space.reactant_list

    @property
    def action_dim(self):
        return self.action_space.spaces[1].shape[0]

    def query_top_k(self, query_array, reactant_idx_list):
        mol_array = self.reactant_array[torch.tensor(reactant_idx_list, device=self.device)]
        dist_array = torch.linalg.norm(mol_array - query_array, dim=1)
        #top_k_idx = dist_array.topk(self.n_lookup_reactants, largest=False)[1].tolist()
        top_k_idx = torch.argsort(dist_array, descending=False).tolist()
        return [reactant_idx_list[each_idx] for each_idx in top_k_idx]

    def step(self, action):
        '''
        '''
        if action != self.action_space.stop_action:
            if action[1].shape != (self.action_dim,):
                raise ValueError('action[1] must be in shape {}'.format((self.action_dim,)))

        done = self.n_step >= self.max_step
        self.terminal = int((action == self.action_space.stop_action) or self.terminal == 1)

        # get product_mol
        if done or self.terminal:
            product_mol = self.current_mol
        else:
            reaction_idx = torch.where(action[0] == 1.0)[0].item()
            reaction = self.action_space.reaction_corpus.reaction_list[reaction_idx]
            if reaction.n_reactant == 1:
                product_mol_candidates = reaction.apply_to([self.current_mol])
                if product_mol_candidates:
                    product_mol = self.rng.choice(product_mol_candidates)[0]
                else:
                    product_mol = self.current_mol
            elif reaction.n_reactant == 2:
                product_mol_list = []
                for each_position in range(reaction.n_reactant):
                    try:
                        legal_reactants = self.action_space.reaction_corpus.legal_reactants(
                            reaction_idx=reaction_idx,
                            current_mol=self.current_mol,
                            current_mol_pos=each_position)
                    except ValueError:
                        continue
                    legal_reactant_idx_list = legal_reactants[1-each_position]
                    if not legal_reactant_idx_list:
                        continue
                    top_k_reactant_idx_list = self.query_top_k(
                        action[1],
                        legal_reactant_idx_list)
                    for each_reactant_idx in top_k_reactant_idx_list:
                        each_reactant = self.action_space.reaction_corpus.reactant_list[each_reactant_idx]
                        if each_position == 0:
                            product_mol_candidates = reaction.apply_to(
                                [self.current_mol, each_reactant])
                        else:
                            product_mol_candidates = reaction.apply_to(
                                [each_reactant, self.current_mol])
                        if product_mol_candidates:
                            product_mol_list \
                                = product_mol_list + list(product_mol_candidates)
                            break
                try:
                    product_mol = self.rng.choice(product_mol_list)[0]
                except:
                    product_mol = self.current_mol
            else:
                raise ValueError

        # compute reward
        reward = self.reward_model(self.observation,
                                   action,
                                   done,
                                   product_mol)

        # update internal states
        self.current_mol = product_mol
        self.n_step += 1
        return self.observation, reward, done, {}

    def reward_model(self, observation, action, done, product_mol):
        # if done, product_mol == current_mol
        reward = self.qsar2reward(product_mol)
        return float(reward)


class DiscreteActionReactionEnv(ReactionEnvBase):

    def __init__(self,
                 seed,
                 reactant_path=None,
                 reaction_tsv_path=None,
                 qsar_model=None,
                 opt_orientation='maximization',
                 max_step=10,
                 penalty_reward=-10.0,
                 allow_multiple_matches=False,
                 device='cpu',
                 **kwargs):
        self.device = device
        self.set_seed(seed)
        action_space = DiscreteReactionSpace(reactant_path, reaction_tsv_path,
                                             seed=self.gen_seed(),
                                             max_reactant=2,
                                             allow_multiple_matches=allow_multiple_matches,
                                             device=device)
        super().__init__(seed=self.gen_seed(),
                         reactant_path=reactant_path,
                         reaction_tsv_path=reaction_tsv_path,
                         qsar_model=qsar_model,
                         opt_orientation=opt_orientation,
                         max_step=max_step,
                         action_space=action_space)
        self.penalty_reward = penalty_reward

    def to(self, device):
        self.device = device
        self.action_space.to(device)
        self.qsar_model.to(device)
        return self

    @property
    def reactant_list(self):
        return self.observation_space.reactant_list

    def step(self, action):
        '''
        '''

        done = self.n_step >= self.max_step
        self.terminal = int((action == self.action_space.stop_action) or self.terminal == 1)

        # get product_mol
        if done or self.terminal:
            product_mol = self.current_mol
        else:
            reaction_idx, reactant_idx = action

            reaction = self.action_space.reaction_corpus.reaction_list[reaction_idx]
            if reaction.n_reactant == 1:
                product_mol_candidates = reaction.apply_to([self.current_mol])
                if product_mol_candidates:
                    product_mol = self.rng.choice(product_mol_candidates)[0]
                else:
                    product_mol = self.current_mol
            elif reaction.n_reactant == 2:
                product_mol_list = []
                for each_position in range(reaction.n_reactant):
                    reactant = self.action_space.reaction_corpus.reactant_list[reactant_idx]
                    if each_position == 0:
                        product_mol_candidates = reaction.apply_to(
                            [self.current_mol, reactant])
                    else:
                        product_mol_candidates = reaction.apply_to(
                            [reactant, self.current_mol])
                    if product_mol_candidates:
                        product_mol_list \
                            = product_mol_list + list(product_mol_candidates)
                        break
                try:
                    product_mol = self.rng.choice(product_mol_list)[0]
                except:
                    product_mol = self.current_mol
            else:
                raise ValueError

        # compute reward
        reward = self.reward_model(self.observation,
                                   action,
                                   done,
                                   product_mol)

        # update internal states
        self.current_mol = product_mol
        self.n_step += 1
        return self.observation, reward, done, {}

    def reward_model(self, observation, action, done, product_mol):
        # if done, product_mol == current_mol
        reward = self.qsar2reward(product_mol)
        return float(reward)


# class AlwaysRewardReactionEnv(ReactionEnvBase):

#     ''' each step, the agent is rewarded as much as qsar score to be maximized.
#     if the agent fails to update the current molecule, it will be punished.
#     '''

#     penalty_reward = -10

#     def reward_model(self, current_mol, action, done, product_mol):
#         reward = 0
#         if not done:
#             if current_mol.HasSubstructMatch(product_mol) \
#                and product_mol.HasSubstructMatch(current_mol):
#                 reward = self.penalty_reward
#             else:
#                 reward = self.qsar2reward(product_mol)
#         return reward


class GoalRewardReactionEnv(ReactionEnvBase):

    ''' At the final step, the agent is rewarded as much as qsar score.
    if the agent fails to update the current molecule, it will be punished.
    '''

    penalty_reward = -10

    def reward_model(self, observation, action, done, product_mol):
        current_mol = observation['current_mol']
        if done:
            # if done, product_mol == current_mol
            try:
                reward = self.qsar2reward(current_mol)
            except:
                reward = self.penalty_reward
        else:
            '''
            if current_mol.HasSubstructMatch(product_mol) \
               and product_mol.HasSubstructMatch(current_mol):
                reward = self.penalty_reward
            else:
            '''
            reward = 0
        return float(reward)


class GoalRewardContinuousActionReactionEnv(ContinuousActionReactionEnv):

    penalty_reward = -10

    def reward_model(self, observation, action, done, product_mol):
        if done:
            # if done, product_mol == current_mol
            try:
                reward = self.qsar2reward(product_mol)
            except:
                reward = self.penalty_reward
        else:
            reward = 0.
        return float(reward)


class GoalRewardDiscreteActionReactionEnv(DiscreteActionReactionEnv):

    penalty_reward = -10

    def reward_model(self, observation, action, done, product_mol):
        if done:
            # if done, product_mol == current_mol
            try:
                reward = self.qsar2reward(product_mol)
            except:
                reward = self.penalty_reward
        else:
            reward = 0.
        return float(reward)
