#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2020"
__version__ = "0.1"

from pathlib import Path
import unittest
import pandas as pd
from rdkit import Chem
from molgym.chem.fingerprint import DenseMorganFingerprint
from molgym.chem.qsar import (CCR5,
                              HIVInt,
                              HIVRT,
                              LinearDenseQSAR)


class TestQSAR(unittest.TestCase):

    def setUp(self):
        mol_df = pd.read_csv(str(Path(__file__).parent.parent / Path('ChEBL.csv'))).loc[
            :, ['SMILES', 'pChEMBL Value']]
        self.mol_list = [Chem.MolFromSmiles(each_smiles) for each_smiles in mol_df.iloc[:, 0]]
        self.tgt_list = mol_df.iloc[:, 1].tolist()

    def test_apollo(self):
        mol = Chem.MolFromSmiles('c1ccccc1')
        for each_cls in [CCR5, HIVInt, HIVRT]:
            pred = each_cls()
            print(pred.forward(mol))

    def test_linear(self):
        fingerprint = DenseMorganFingerprint()
        model = LinearDenseQSAR(fingerprint, 'Ridge', {'alpha': 1.0})
        model.fit(self.mol_list, self.tgt_list)
        print(model.forward(self.mol_list[0]))
