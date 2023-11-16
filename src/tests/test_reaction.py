#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

import os
import unittest
from rdkit import Chem
from molgym.chem.reaction import Reaction, ReactionCorpus


class TestReaction(unittest.TestCase):
    def setUp(self):
        self.reaction_name = 'amid bond formation'
        self.reaction_smarts = '[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]'
        self.reactant_list = [Chem.MolFromSmiles('CC(=O)O'),
                              Chem.MolFromSmiles('NC')]

    def test_init(self):
        reaction = Reaction(self.reaction_smarts, self.reaction_name)

    def test_is_reactant(self):
        reaction = Reaction(self.reaction_smarts, self.reaction_name)
        self.assertTrue(reaction.is_reactant(self.reactant_list[0]))
        self.assertTrue(reaction.is_reactant(self.reactant_list[1]))

    def test_apply(self):
        reaction = Reaction(self.reaction_smarts, self.reaction_name)
        possible_product_list = reaction.apply_to(self.reactant_list)
        self.assertEqual(Chem.MolToSmiles(possible_product_list[0][0]), 'CNC(C)=O')


class TestReactionCorpus(unittest.TestCase):

    def setUp(self):
        self.reaction_corpus = ReactionCorpus(
            reactant_path=os.path.join(os.path.dirname(__file__), 'building_blocks.csv'),
            reaction_tsv_path=os.path.join(os.path.dirname(__file__), 'rxn.tsv'),
            max_reactant=2)

    def test_legal_reaction_template(self):
        current_mol = self.reaction_corpus.reactant_list[0]
        legal_idx_list = self.reaction_corpus.legal_reaction_template(current_mol)
        reaction_idx, current_mol_pos = legal_idx_list[0]
        reactant_list = self.reaction_corpus.legal_reactants(
            reaction_idx=reaction_idx,
            current_mol=current_mol,
            current_mol_pos=current_mol_pos)
        reactants = []
        for each_idx in range(len(reactant_list)):
            if each_idx == current_mol_pos:
                reactants.append(reactant_list[each_idx][0])
            else:
                reactants.append(self.reaction_corpus.reactant_list[reactant_list[each_idx][0]])
        reaction = self.reaction_corpus.reaction_list[reaction_idx]
        product_mol_candidates = reaction.apply_to(reactants)
        print('product:', Chem.MolToSmiles(product_mol_candidates[0][0]))


if __name__ == '__main__':
    unittest.main()
