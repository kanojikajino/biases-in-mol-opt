#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

import os
import unittest
from rdkit import Chem
from molgym.rl.space import ReactionSpace, ContinuousReactionSpace


class TestReactionSpace(unittest.TestCase):

    def test_legal(self):
        self.action_space = ReactionSpace(
            os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            os.path.join(os.path.dirname(__file__), 'rxn.tsv'))

        current_mol = Chem.MolFromSmiles('C#CCC1(C(=O)O)CCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1')
        action_dict = self.action_space.legal_sample({'current_mol': current_mol,
                                                      'n_step': 0})
        reactant_list = []
        for each_idx in action_dict['reactant_list']:
            reactant_list.append(self.action_space.reaction_corpus.reactant_list[each_idx])
        reactant_list.insert(action_dict['current_mol_pos'], current_mol)
        product_list = self.action_space.reaction_corpus.reaction_list[action_dict['reaction_idx']]\
                                        .apply_to(reactant_list)
        self.assertTrue(product_list)

    def test_continuous(self):
        self.action_space = ContinuousReactionSpace(
            low=-1.0,
            high=1.0,
            action_dim=10,
            reactant_path=os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            reaction_tsv_path=os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            max_reactant=2)
        current_mol = Chem.MolFromSmiles('C#CCC1(C(=O)O)CCN(C(=O)OCC2c3ccccc3-c3ccccc32)CC1')
        print('reaction_idx = {}, reactant_query = {}'.format(
            *self.action_space.legal_sample({'current_mol': current_mol,
                                             'n_step': 0})))


if __name__ == '__main__':
    unittest.main()
