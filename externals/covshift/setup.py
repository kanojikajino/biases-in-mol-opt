#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' setup file for covshift. '''

__author__ = 'Hiroshi Kajino'
__copyright__ = '(C) Copyright IBM Corp. 2021'

from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='covshift',
    version='0.1',
    author='Hiroshi Kajino',
    author_email='hiroshi.kajino.1989@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    test_suite='tests',
    include_package_data=True,
    install_requires=_requires_from_file('requirements.txt'),
)
