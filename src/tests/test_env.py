#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

import os
import unittest
import torch
from rdkit import Chem
from molgym.rl.env import GoalRewardReactionEnv, ContinuousActionReactionEnv
from molgym.chem.qsar import PenalizedLogP
from molgym.chem.fingerprint import DenseMorganFingerprint


class TestMolecularReactionEnv(unittest.TestCase):

    def test_step(self):
        qsar_model = PenalizedLogP()
        env = GoalRewardReactionEnv(
            reactant_path=os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            reaction_tsv_path=os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            qsar_model=qsar_model,
            max_step=4)
        env.set_seed(2)
        env.reset()
        print('\n')
        print(env.n_step, env.current_smiles)
        for _ in range(5):
            obs, reward, done, info = env.step({'stop': 0,
                                                'reaction_idx': 0,
                                                'current_mol_pos': 0,
                                                'reactant_list': [0]})
            print(env.n_step, Chem.MolToSmiles(obs['current_mol']), reward, done, info)


class TestContinuousActionReactionEnvBase(unittest.TestCase):

    def test_step(self):
        qsar_model = PenalizedLogP()
        fingerprint = DenseMorganFingerprint()
        env = ContinuousActionReactionEnv(
            n_lookup_reactants=3,
            low=0.0,
            high=1.0,
            fingerprint=fingerprint,
            reactant_path=os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            reaction_tsv_path=os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            qsar_model=qsar_model,
            max_step=4)
        env.set_seed(2)
        env.reset()
        print('\n')
        print(env.n_step, env.current_smiles)
        action = list(env.action_space.sample())
        action[0] = torch.zeros(env.action_space[0].n)
        action[0][0] = 1.0
        for _ in range(5):
            obs, reward, done, info = env.step(action)
            print(env.n_step, Chem.MolToSmiles(obs['current_mol']), reward, done, info)
        

if __name__ == '__main__':
    unittest.main()
