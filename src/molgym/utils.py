#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2021"

import math
from rdkit import Chem
import pandas as pd


def construct_dataset(file_path=None, mol_col=None, tgt_col=None, preprocessing_list=[]):
    if file_path:
        mol_df = pd.read_csv(file_path).loc[
            :, [mol_col, tgt_col]]
        mol_list = [Chem.MolFromSmiles(each_smiles) for each_smiles in mol_df.iloc[:, 0]]
        tgt_list = mol_df.iloc[:, 1].tolist()
    else:
        mol_list = []
        tgt_list = []
    for each_preprocessor in preprocessing_list:
        tgt_list = [eval(each_preprocessor)(each_tgt) for each_tgt in tgt_list]
    return mol_list, tgt_list


def mol2hash(mol):
    return Chem.MolToInchiKey(mol)
