#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#rxn = AllChem.ReactionFromSmarts('[C:1]=[C:2][C:3]=[C:4].[C:5]=[C:6]>>[C:1]1=[C:2][C:3]=[C:4][C:5]=[C:6]1')
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')
reactant1 = Chem.MolFromSmiles('CC(=O)O')
reactant2 = Chem.MolFromSmiles('NC')
product = rxn.RunReactants((reactant1, reactant2))[0][0]
print(Chem.MolToSmiles(product))
rxn_instance = AllChem.ReactionFromSmarts(Chem.MolToSmiles(reactant1) + '.'\
                                          + Chem.MolToSmiles(reactant2) + '>>'\
                                          + Chem.MolToSmiles(product))
d2d = Draw.MolDraw2DCairo(800, 300)
d2d.DrawReaction(rxn_instance, highlightByReactant=True)
png = d2d.GetDrawingText()
open('reaction.png', 'wb+').write(png)

# Benzene
Draw.MolToFile(Chem.MolFromSmiles('c1ccccc1'), 'benzene.png')
