#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' setup file for mlbasics '''

__author__ = 'Hiroshi Kajino'
__copyright__ = '(C) Copyright IBM Corp. 2022'

from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='mlbasics',
    version='0.1',
    author='Hiroshi Kajino',
    author_email='hiroshi.kajino.1989@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    test_suite='tests',
    include_package_data=True,
    package_data={'': ['*.gz', '*.pkl', '*.pklz']},
    install_requires=_requires_from_file('requirements.txt'),
)
