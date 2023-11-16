#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2021'

import unittest
from rdkit import Chem
from molgym.chem.fingerprint import SparseMorganFingerprint, SelectedMolecularDescriptorsFingerprint


class TestFingerprint(unittest.TestCase):

    def test_fingerprint(self):
        bit = 4096
        morgan_fingerprint = SparseMorganFingerprint(bit=bit)
        fp = morgan_fingerprint.forward(Chem.MolFromSmiles('c1ccccc1'))
        for each_element in fp:
            self.assertLess(each_element, bit)

class TestMolecularDescriptorsFingerprint(unittest.TestCase):

    def test_fingerprint(self):
        fingerprint = SelectedMolecularDescriptorsFingerprint()
        print(fingerprint.forward(Chem.MolFromSmiles('c1ccccc1')))
