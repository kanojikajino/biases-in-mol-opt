#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' setup file for apollo1060 '''

__author__ = 'Hiroshi Kajino'
__copyright__ = '(C) Copyright IBM Corp. 2021'

from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='apollo1060',
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    test_suite='tests',
    include_package_data=True,
    package_data={'': ['*.json', '*.csv', '*.npy', '*.sav', '*.txt']},
    install_requires=_requires_from_file('requirements.txt'),
)
