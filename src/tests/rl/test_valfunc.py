#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2021'

from pathlib import Path
import unittest
import torch
from molgym.rl.valfunc import (TwoStageActionValueFunction,
                               LinearSparseActionValueFunction)
from molgym.rl.agent import LegalRandomAgent
from molgym.rl.env import GoalRewardReactionEnv
from molgym.chem.qsar import PenalizedLogP
from molgym.ml.linear import LinearSparsePredictor
from molgym.chem.fingerprint import SparseMorganFingerprint
from molgym.rl.feature import SingleSparseFingerprintFeature
from molgym.rl.episode import EpisodeMemory


class TestActionValueFunction(unittest.TestCase):

    def setUp(self):
        plog = PenalizedLogP()
        self.env = GoalRewardReactionEnv(
            str(Path(__file__).parent.parent / Path('building_blocks.csv')),
            str(Path(__file__).parent.parent / Path('rxn.tsv')),
            qsar_model=plog)
        self.env.set_seed(4)
        self.agent = LegalRandomAgent(action_space=self.env.action_space)

    def test_fingerprint_action_value(self):
        fingerprint = SparseMorganFingerprint()
        state_action_feature = SingleSparseFingerprintFeature(fingerprint)
        predictor = LinearSparsePredictor(in_dim=state_action_feature.out_dim)
        action_val_func = TwoStageActionValueFunction(
            predictor=predictor,
            state_action_feature=state_action_feature)
        obs = self.env.reset()
        action_dict = self.agent.act(obs, self.env.action_space)
        print(action_val_func.forward(obs, action_dict, self.env.action_space))

    def test_zdd(self):
        zdd_fingerprint = SparseMorganFingerprint(use_zdd=True)
        no_zdd_fingerprint = SparseMorganFingerprint(use_zdd=False)
        zdd_state_action_feature = SingleSparseFingerprintFeature(zdd_fingerprint)
        no_zdd_state_action_feature = SingleSparseFingerprintFeature(no_zdd_fingerprint)

        zdd_action_val_func = LinearSparseActionValueFunction(zdd_state_action_feature)
        no_zdd_action_val_func = LinearSparseActionValueFunction(no_zdd_state_action_feature)
        zdd_action_val_func.predictor.emb.weight.data \
            = no_zdd_action_val_func.predictor.emb.weight.data

        behavior_agent = LegalRandomAgent(action_space=self.env.action_space)

        for _ in range(10):
            episode_list = []
            for _ in range(10):
                episode_list.append(behavior_agent.gen_episode(self.env))
            episode_memory = EpisodeMemory(episode_list, seed=46)
            zdd_action_val_func.state_action_feature.fingerprint.prep(
                episode_memory,
                Path(__file__).parent,
                self.env.action_space)
            for each_transition in episode_memory.transition_list:
                self.assertTrue(
                    torch.allclose(
                        zdd_action_val_func.greedy_predict(each_transition.state,
                                                           self.env.action_space),
                        no_zdd_action_val_func.greedy_predict(each_transition.state,
                                                              self.env.action_space)))
                try:
                    self.assertTrue(
                        torch.allclose(
                            zdd_action_val_func.greedy_predict(each_transition.next_state,
                                                               self.env.action_space),
                            no_zdd_action_val_func.greedy_predict(each_transition.next_state,
                                                                  self.env.action_space)))
                except:
                    import pdb; pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
