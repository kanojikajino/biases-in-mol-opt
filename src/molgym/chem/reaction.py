#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from joblib import Parallel, delayed
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions


class Reaction:

    ''' Chemical reaction class
    '''

    def __init__(self, reaction_smarts, reaction_name, allow_multiple_matches=False):
        self.reaction_name = reaction_name
        self.reaction_smarts = reaction_smarts
        self.reaction_obj = AllChem.ReactionFromSmarts(reaction_smarts)
        self.allow_multiple_matches = allow_multiple_matches
        rdChemReactions.ChemicalReaction.Initialize(self.reaction_obj)
        # the followings are...
        #rdChemReactions.PreprocessReaction(self.reaction_obj)
        #rdChemReactions.SanitizeRxn(self.reaction_obj)

    @property
    def n_agent(self):
        return self.reaction_obj.GetNumAgentTemplates()

    @property
    def n_reactant(self):
        return self.reaction_obj.GetNumReactantTemplates()

    @property
    def n_product(self):
        return self.reaction_obj.GetNumProductTemplates()

    def is_applicable(self, mol, position):
        ''' return whether this reaction is applicable when `mol` is used as `position`-th reactant.
        '''
        if not self.reaction_obj.IsMoleculeReactant(mol):
            return False
        if self.allow_multiple_matches:
            return len(mol.GetSubstructMatches(self.get_reactant_template(position),
                                               useChirality=True,
                                               maxMatches=1)) != 0
        else:
            return len(mol.GetSubstructMatches(self.get_reactant_template(position),
                                               useChirality=True,
                                               maxMatches=2)) == 1

    def get_reactant_template(self, position):
        ''' get reactant template (substracture that the reactant must have)
        at `position` of this reaction.

        Parameters
        ----------
        position : int
            position of the reactant

        Returns
        -------
        rdkit.Chem.rdchem.Mol
        '''
        if not position < self.n_reactant:
            raise ValueError('position {} must be smaller than {}'.format(position,
                                                                          self.n_reactant))
        return self.reaction_obj.GetReactantTemplate(position)

    def is_reactant(self, mol):
        ''' check whether the input mol is a reactant of this reaction.

        Parameters
        ----------
        mol : Mol

        Returns
        -------
        bool
            whether or not the input molecule is a reactant
        '''
        _is_reactant = False
        if mol is not None:
            _is_reactant = self.reaction_obj.IsMoleculeReactant(mol)
        return _is_reactant

    def apply_to(self, mol_list):
        ''' apply this reaction to `mol_list`
        note that the resultant products are dependent on the order in `mol_list`,
        because `reaction_smarts` specifies the order.

        Parameters
        ----------
        mol_list : List

        Returns
        -------
        list of lists of products
        '''
        '''
        possible_product_list = list(self.reaction_obj.RunReactants(mol_list))
        for each_product_list in possible_product_list:
            for each_prod in each_product_list:
                Chem.SanitizeMol(each_prod)
        return possible_product_list
        '''
        return self._apply_to(tuple(mol_list))

    def _apply_to(self, mol_tuple):
        possible_product_list = list(self.reaction_obj.RunReactants(mol_tuple))
        return_list = []
        for each_product_list in possible_product_list:
            each_return_list = []
            success = True
            for each_prod in each_product_list:
                try:
                    Chem.SanitizeMol(each_prod)
                    each_return_list.append(each_prod)
                except:
                    success = False
            if success:
                return_list.append(each_return_list)
        return return_list


class ReactionCorpus:

    ''' this class maintins a list of reaction templates and a list of reactants.


    '''

    def __init__(self, reactant_path, reaction_tsv_path, max_reactant, allow_multiple_matches=False):
        self.reactant_list = [each_mol for each_mol \
                              in Chem.SmilesMolSupplier(reactant_path, titleLine=False)]
        self.reaction_list = []
        self.max_reactant = 0
        df = pd.read_csv(reaction_tsv_path, delimiter='\t', header=None, names=['name', 'smarts'])
        for _, each_entry in df.iterrows():
            each_reaction = Reaction(each_entry['smarts'], each_entry['name'], allow_multiple_matches)
            if each_reaction.n_reactant <= max_reactant:
                self.reaction_list.append(each_reaction)
                self.max_reactant = max(self.max_reactant, self.reaction_list[-1].n_reactant)
        self.n_reactant = len(self.reactant_list)
        self.n_reaction = len(self.reaction_list)
        self.init_legal_reactant()

    def init_legal_reactant(self):
        ''' for each reaction template, list up all legal reactants

        self.legal_building_block_id_list[reaction_idx] returns
        the list of length `num_reactants`.
        self.legal_building_block_id_list[reaction_idx][each_position_idx] returns
        the list of reactants applicable to that position.
        '''
        def _init_sub_legal_reactant(each_reaction, reactant_list):
            n_reactants = each_reaction.n_reactant
            position_wise_legal_reactant = []
            for each_position in range(n_reactants):
                each_list = []
                for each_idx, each_reactant in enumerate(reactant_list):
                    if each_reaction.is_applicable(each_reactant, each_position):
                        each_list.append(each_idx)
                position_wise_legal_reactant.append(each_list)
            return position_wise_legal_reactant

        self.legal_reactant_list = Parallel(n_jobs=-1, verbose=10)(
            [delayed(_init_sub_legal_reactant)(each_reaction, self.reactant_list)
             for each_reaction in self.reaction_list])

    def legal_reaction_template(self, current_mol):
        ''' return indices of legal reaction templates and positions for `current_mol`
        '''
        legal_idx_list = []
        for each_reaction_idx, each_reaction in enumerate(self.reaction_list):
            for each_position in range(each_reaction.n_reactant):
                if each_reaction.is_applicable(current_mol, each_position):
                    legal_idx_list.append((each_reaction_idx, each_position))
        return legal_idx_list

    def legal_reactants(self, reaction_idx, current_mol, current_mol_pos):
        ''' return list of reactants applicable to the reaction with `reaction_idx`,
        when `current_mol` is used as the `current_mol_pos`-th reactants.
        '''
        if not self.reaction_list[reaction_idx].is_applicable(
                current_mol,
                current_mol_pos):
            raise ValueError('this reaction cannot be applied to current_mol at {}-th reactant'\
                             .format(current_mol_pos))
        reaction = self.reaction_list[reaction_idx]
        legal_reactant_list = self.legal_reactant_list[reaction_idx]
        reactant_list = []
        for each_mol_pos in range(reaction.n_reactant):
            if each_mol_pos == current_mol_pos:
                reactant_list.append([current_mol])
            else:
                reactant_list.append(legal_reactant_list[each_mol_pos])
        return reactant_list

    def legal_reaction(self, current_mol):
        ''' return possible reactions for `current_mol`

        Parameters
        ----------
        current_mol: Mol

        Returns
        -------
        list of dict
            each dict contains legal reaction information.
            - `reaction_idx`: the index of the reaction applicable to `current_mol`
            - `current_mol_pos`: the position of `current_mol` in this reaction template
            - `candidate_reactant`: list
                `each_position_idx`-th element contains the list of reactants applicable to that position.
        '''
        return_list = []
        for each_reaction_idx, each_reaction in enumerate(self.reaction_list):
            each_match_list = []
            for each_position in range(each_reaction.n_reactant):
                if each_reaction.is_applicable(current_mol, each_position):
                    feasible = True
                    for each_other_position in range(each_reaction.n_reactant):
                        if each_other_position != each_position:
                            feasible = feasible \
                                and self.legal_reactant_list[each_reaction_idx][each_other_position]
                    if feasible:
                        each_match_list.append(each_position)
            if each_match_list:
                each_dict = {
                    'reaction_idx': each_reaction_idx,
                    'current_mol_pos': each_match_list,
                    'candidate_reactant': self.legal_reactant_list[each_reaction_idx]}
                return_list.append(each_dict)
        return return_list
