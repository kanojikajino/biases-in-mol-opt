#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2021"


from apollo1060.pipelines import ClassicPipe
import apollo1060.models.hiv_ccr5
import apollo1060.models.hiv_int
import apollo1060.models.hiv_rt


ccr5_pipe = ClassicPipe.load(apollo1060.models.hiv_ccr5.__path__[0])
int_pipe = ClassicPipe.load(apollo1060.models.hiv_int.__path__[0])
rt_pipe = ClassicPipe.load(apollo1060.models.hiv_rt.__path__[0])
