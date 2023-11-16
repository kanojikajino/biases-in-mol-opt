##!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'


# set luigi_config_path BEFORE importing luigi
from datetime import datetime
import glob
import logging
import os
from pathlib import Path
import sys
import luigi

from luigine.abc import (AutoNamingTask,
                         main)
from molgym.engines.main import (PerformanceEvaluation,
                                 RandomPerformanceEvaluation,
                                 MultiplePerformanceEvaluation,
                                 MultipleRandomPerformanceEvaluation,
                                 MultipleRun,
                                 PlotBiasEstimation,
                                 EvaluateComputationTime)

try:
    working_dir = Path(sys.argv[1:][sys.argv[1:].index("--working-dir")
                                    + 1]).resolve()
except ValueError:
    raise ValueError("--working-dir option must be specified.")

# load parameters from `INPUT/param.py`
sys.path.append(str((working_dir / 'INPUT').resolve()))
from param import (
    ReactantSelector_params,
    OfflineTrainDataConstruction_params,
    OfflineTestDataConstruction_params,
    OfflineDataPreprocessing_params,
    EnvSetup_params,
    EnvTrainTestSetup_params,
    FeatureSetup_params,
    AgentSetup_params,
    Train_params,
    RunTest_params,
    TrainRewardModel4Eval_params,
    PerformanceEvaluation_params,
    MultipleRun_params,
    PlotBiasEstimation_params)

# setup logger
logger = logging.getLogger('luigi-interface')

# redirect stdout and stderr to logger
class StreamToLogger(object):
    '''
    Fake file-like stream object that redirects writes to a logger instance.
    '''
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

logger_sys = logging.getLogger('luigi-interface.stdout')
#sys.stdout = StreamToLogger(logger_sys, logging.INFO)
#sys.stderr = StreamToLogger(logger_sys, logging.INFO)

AutoNamingTask._working_dir = working_dir
AutoNamingTask.working_dir = luigi.Parameter(default=str(working_dir))
# ----------- preamble ------------

# Define tasks



class PerformanceEvaluation(PerformanceEvaluation):

    ReactantSelector_params = luigi.DictParameter(
        default=ReactantSelector_params)
    EnvSetup_params = luigi.DictParameter(
        default=EnvSetup_params)
    OfflineTrainDataConstruction_params = luigi.DictParameter(
        default=OfflineTrainDataConstruction_params)
    OfflineTestDataConstruction_params = luigi.DictParameter(
        default=OfflineTestDataConstruction_params)
    OfflineTrainDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    OfflineTestDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    EnvTrainTestSetup_params = luigi.DictParameter(
        default=EnvTrainTestSetup_params)
    FeatureSetup_params = luigi.DictParameter(
        default=FeatureSetup_params)
    AgentSetup_params = luigi.DictParameter(
        default=AgentSetup_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
    RunTest_params = luigi.DictParameter(
        default=RunTest_params)
    TrainRewardModel4Eval_params = luigi.DictParameter(
        default=TrainRewardModel4Eval_params)
    PerformanceEvaluation_params = luigi.DictParameter(
        default=PerformanceEvaluation_params)

class RandomPerformanceEvaluation(RandomPerformanceEvaluation):

    ReactantSelector_params = luigi.DictParameter(
        default=ReactantSelector_params)
    EnvSetup_params = luigi.DictParameter(
        default=EnvSetup_params)
    OfflineTrainDataConstruction_params = luigi.DictParameter(
        default=OfflineTrainDataConstruction_params)
    OfflineTestDataConstruction_params = luigi.DictParameter(
        default=OfflineTestDataConstruction_params)
    OfflineTrainDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    OfflineTestDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    EnvTrainTestSetup_params = luigi.DictParameter(
        default=EnvTrainTestSetup_params)
    FeatureSetup_params = luigi.DictParameter(
        default=FeatureSetup_params)
    AgentSetup_params = luigi.DictParameter(
        default=AgentSetup_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
    RunTest_params = luigi.DictParameter(
        default=RunTest_params)
    TrainRewardModel4Eval_params = luigi.DictParameter(
        default=TrainRewardModel4Eval_params)
    PerformanceEvaluation_params = luigi.DictParameter(
        default=PerformanceEvaluation_params)

class MultiplePerformanceEvaluation(MultiplePerformanceEvaluation):

    ReactantSelector_params = luigi.DictParameter(
        default=ReactantSelector_params)
    EnvSetup_params = luigi.DictParameter(
        default=EnvSetup_params)
    OfflineTrainDataConstruction_params = luigi.DictParameter(
        default=OfflineTrainDataConstruction_params)
    OfflineTestDataConstruction_params = luigi.DictParameter(
        default=OfflineTestDataConstruction_params)
    OfflineTrainDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    OfflineTestDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    EnvTrainTestSetup_params = luigi.DictParameter(
        default=EnvTrainTestSetup_params)
    FeatureSetup_params = luigi.DictParameter(
        default=FeatureSetup_params)
    AgentSetup_params = luigi.DictParameter(
        default=AgentSetup_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
    RunTest_params = luigi.DictParameter(
        default=RunTest_params)
    TrainRewardModel4Eval_params = luigi.DictParameter(
        default=TrainRewardModel4Eval_params)
    PerformanceEvaluation_params = luigi.DictParameter(
        default=PerformanceEvaluation_params)
    seed = luigi.IntParameter(default=43)
    n_iter = luigi.IntParameter(default=1)


class MultipleRandomPerformanceEvaluation(MultipleRandomPerformanceEvaluation):

    ReactantSelector_params = luigi.DictParameter(
        default=ReactantSelector_params)
    EnvSetup_params = luigi.DictParameter(
        default=EnvSetup_params)
    OfflineTrainDataConstruction_params = luigi.DictParameter(
        default=OfflineTrainDataConstruction_params)
    OfflineTestDataConstruction_params = luigi.DictParameter(
        default=OfflineTestDataConstruction_params)
    OfflineTrainDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    OfflineTestDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    EnvTrainTestSetup_params = luigi.DictParameter(
        default=EnvTrainTestSetup_params)
    FeatureSetup_params = luigi.DictParameter(
        default=FeatureSetup_params)
    AgentSetup_params = luigi.DictParameter(
        default=AgentSetup_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
    RunTest_params = luigi.DictParameter(
        default=RunTest_params)
    TrainRewardModel4Eval_params = luigi.DictParameter(
        default=TrainRewardModel4Eval_params)
    PerformanceEvaluation_params = luigi.DictParameter(
        default=PerformanceEvaluation_params)
    seed = luigi.IntParameter(default=43)
    n_iter = luigi.IntParameter(default=1)

class MultipleRun(MultipleRun):

    MultipleRun_params = luigi.DictParameter(default=MultipleRun_params)


class PlotBiasEstimation(PlotBiasEstimation):

    MultipleRun_params = luigi.DictParameter(default=MultipleRun_params)
    LinePlotMultipleRun_params = luigi.DictParameter(default=PlotBiasEstimation_params)


class EvaluateComputationTime(EvaluateComputationTime):

    ReactantSelector_params = luigi.DictParameter(
        default=ReactantSelector_params)
    EnvSetup_params = luigi.DictParameter(
        default=EnvSetup_params)
    OfflineTrainDataConstruction_params = luigi.DictParameter(
        default=OfflineTrainDataConstruction_params)
    OfflineTestDataConstruction_params = luigi.DictParameter(
        default=OfflineTestDataConstruction_params)
    OfflineTrainDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    OfflineTestDataPreprocessing_params = luigi.DictParameter(
        default=OfflineDataPreprocessing_params)
    EnvTrainTestSetup_params = luigi.DictParameter(
        default=EnvTrainTestSetup_params)
    FeatureSetup_params = luigi.DictParameter(
        default=FeatureSetup_params)
    AgentSetup_params = luigi.DictParameter(
        default=AgentSetup_params)
    Train_params = luigi.DictParameter(
        default=Train_params)
        

if __name__ == "__main__":
    for each_engine_status in glob.glob(str(working_dir / 'engine_status.*')):
        os.remove(each_engine_status)
    with open(working_dir / 'engine_status.ready', 'w') as f:
        f.write("ready: {}\n".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
    main(working_dir)
