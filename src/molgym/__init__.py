#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

from gym.envs.registration import register

register(
    id='molenv-v0',
    entry_point='molgym.env:MolecularReactionEnv'
)
