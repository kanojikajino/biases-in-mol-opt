#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

import os
import unittest
from rdkit import Chem
from molgym.rl.agent import LegalRandomAgent
from molgym.rl.env import GoalRewardReactionEnv, ContinuousActionReactionEnv
from molgym.chem.qsar import PenalizedLogP
from molgym.chem.fingerprint import DenseMorganFingerprint


class TestRandomAgentOnDiscreteEnv(unittest.TestCase):

    def setUp(self):
        plog = PenalizedLogP()
        self.env = GoalRewardReactionEnv(
            os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            qsar_model=plog)
        self.env.set_seed(4)
        self.agent = LegalRandomAgent(action_space=self.env.action_space)
        self.agent.set_seed(43)

    def test_step(self):
        self.env.seed(2)
        self.env.action_space.seed(1)
        obs = self.env.reset()
        reward = 0
        done = False
        print('\n')
        print('{}\t{}'.format('iter', 'smiles'))
        for _ in range(5):
            print('{}\t{}'.format(self.env.n_step,
                                  Chem.MolToSmiles(obs['current_mol'])))
            action_dict = self.agent.act(obs, self.env.action_space)
            if action_dict['stop'] == 1:
                print(' * action: stop')
            else:
                print(' * action: stop={}, reaction_idx={}, current_mol_pos={}, reactant_list={}'.format(
                    action_dict['stop'],
                    action_dict['reaction_idx'],
                    action_dict['current_mol_pos'],
                    action_dict['reactant_list']))
            obs, reward, done, _ = self.env.step(action_dict)
            print(' * env is done: {}'.format(done))
            print(' * reward: {}'.format(reward))
            if done:
                break

    def test_episode(self):
        self.env.seed(2)
        self.env.action_space.seed(1)
        episode = self.agent.gen_episode(self.env)
        next_state = None
        print('\n')
        print(episode)
        for each_transition in episode.transition_list:
            if next_state is not None:
                self.assertTrue(next_state['current_mol'].HasSubstructMatch(each_transition.state['current_mol'])\
                                and each_transition.state['current_mol'].HasSubstructMatch(next_state['current_mol']))
            next_state = each_transition.next_state


class TestRandomAgentOnContinuousEnv(unittest.TestCase):

    def setUp(self):
        qsar_model = PenalizedLogP()
        fingerprint = DenseMorganFingerprint()
        self.env = ContinuousActionReactionEnv(
            n_lookup_reactants=3,
            low=0.0,
            high=1.0,
            fingerprint=fingerprint,
            reactant_path=os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            reaction_tsv_path=os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            qsar_model=qsar_model,
            max_step=4)
        self.env.set_seed(4)
        self.agent = LegalRandomAgent(action_space=self.env.action_space)
        self.agent.set_seed(43)

    def test_step(self):
        self.env.seed(2)
        self.env.action_space.seed(1)
        obs = self.env.reset()
        reward = 0
        done = False
        print('\n')
        print('{}\t{}'.format('iter', 'smiles'))
        for _ in range(5):
            print('{}\t{}'.format(self.env.n_step, Chem.MolToSmiles(obs['current_mol'])))
            action = self.agent.act(obs, self.env.action_space)
            print(' * action: '.format(action))
            obs, reward, done, _ = self.env.step(action)
            print(' * env is done: {}'.format(done))
            print(' * reward: {}'.format(reward))
            if done:
                break

    def test_episode(self):
        self.env.seed(2)
        self.env.action_space.seed(1)
        episode = self.agent.gen_episode(self.env)
        next_state = None
        print('\n')
        print(episode)
        for each_transition in episode.transition_list:
            if next_state is not None:
                self.assertTrue(
                    next_state['current_mol'].HasSubstructMatch(each_transition.state['current_mol'])\
                    and each_transition.state['current_mol'].HasSubstructMatch(next_state['current_mol']))
            next_state = each_transition.next_state


if __name__ == '__main__':
    unittest.main()
