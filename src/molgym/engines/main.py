import logging
import itertools
from joblib import Parallel, delayed
import luigi
from luigi.util import requires
import numpy as np
import pandas as pd
import pathlib
from rdkit import Chem
import shutil
import torch
from luigine.abc import AutoNamingTask, MultipleRunBase, LinePlotMultipleRun

import covshift.weight
import molgym.chem.fingerprint
from mlbasics.utils import RngMixin
from ..rl.episode import EpisodeMemory
from .agent import (RunTest,
                    RunRandomTest,
                    TrainDensityRatio,
                    TrainRandomDensityRatio,
                    Train,
                    FeatureSetup,
                    AgentSetup,
                    TrainRewardModel4Eval,
                    TrainRandomRewardModel4Eval)
from .env import (EnvSetup,
                  EnvTrainTestSetup,
                  EvalPairList)

logger = logging.getLogger('luigi-interface')
AutoNamingTask.output_ext = luigi.Parameter(default='pkl')



def _doubly_robust_estimator(
        train_key,
        test_key,
        test_episode_memory_G1,
        episode_memory_G2,
        predictor_G2=None,
        density_ratio_G1_G2=None):
    def batch_forward(mol_list, predictor):
        if hasattr(predictor, 'batch_forward'):
            return predictor.batch_forward(mol_list)
        return torch.tensor([predictor.forward(each_mol) for each_mol in mol_list])

    est = 0.0
    if predictor_G2:
        plug_in_list = batch_forward(test_episode_memory_G1.done_mol_list, predictor_G2)
        est = est + torch.mean(plug_in_list)
    if density_ratio_G1_G2 is not None:
        density_ratio_G1_G2 = density_ratio_G1_G2.to('cpu')

    if density_ratio_G1_G2 is not None and predictor_G2:
        correction_list = []
        reward_list = []
        mol_list = []
        for each_idx, each_transition in enumerate(episode_memory_G2.done_transition_list):
            reward_list.append(each_transition.reward)
            mol_list.append(each_transition.state['current_mol'])
        correction_list = density_ratio_G1_G2 * (torch.tensor(reward_list) - batch_forward(mol_list, predictor_G2))
        est = est + torch.mean(correction_list)
    elif density_ratio_G1_G2 is not None and not predictor_G2:
        correction_list = []
        for each_idx, each_transition in enumerate(episode_memory_G2.done_transition_list):
            each_mol = each_transition.state['current_mol']
            each_reward = each_transition.reward
            correction_list.append(density_ratio_G1_G2[each_idx] \
                                   * each_reward)
        est = est + torch.mean(torch.tensor(correction_list))
    return (train_key, test_key), float(est)


@requires(EnvTrainTestSetup, RunTest, TrainRewardModel4Eval, EvalPairList)
class PerformanceEvaluation(AutoNamingTask):

    PerformanceEvaluation_params = luigi.DictParameter()

    def run_task(self, input_list):

        train_env, property_tuple_dict, _ = input_list[0]
        episode_list_dict = input_list[1]
        #_, episode_memory_dict = input_list[2]
        qsar_density_ratio_dict = input_list[2]
        eval_pair_list = input_list[3]
        del train_env.action_space

        logger.info(' #### Performance Evaluation ####')
        est_dict = {}
        
        if self.TrainRewardModel4Eval_params['mode'] == 'covariate_shift' and self.PerformanceEvaluation_params['covariate_shift']:
            est_dict = dict(Parallel(
                n_jobs=self.PerformanceEvaluation_params['workers'],
                verbose=10)([
                    delayed(_doubly_robust_estimator)(
                        each_train_key,
                        each_test_key,
                        episode_list_dict[each_train_key],
                        property_tuple_dict[each_test_key].episode_memory, #episode_memory_dict[each_test_key],
                        qsar_density_ratio_dict[(each_train_key, each_test_key)][0] if self.PerformanceEvaluation_params['doubly_robust']['direct_method'] else None, #property_tuple_dict[each_test_key].qsar_model if self.PerformanceEvaluation_params['doubly_robust']['direct_method'] else None,
                        qsar_density_ratio_dict.get(
                            (each_train_key, each_test_key), (None, None))[1] if self.PerformanceEvaluation_params['doubly_robust']['importance_sampling'] else None
                    )
                    for (each_train_key, each_test_key) \
                    in eval_pair_list]))
        else:
            est_dict = dict(Parallel(
                n_jobs=self.PerformanceEvaluation_params['workers'],
                verbose=10)([
                    delayed(_doubly_robust_estimator)(
                        each_train_key,
                        each_test_key,
                        episode_list_dict[each_train_key],
                        property_tuple_dict[each_test_key].episode_memory, #episode_memory_dict[each_test_key],
                        property_tuple_dict[each_test_key].qsar_model if self.PerformanceEvaluation_params['doubly_robust']['direct_method'] else None,
                        qsar_density_ratio_dict.get(
                            (each_train_key, each_test_key), (None, None))[1] if self.PerformanceEvaluation_params['doubly_robust']['importance_sampling'] else None
                    )
                    for (each_train_key, each_test_key) \
                    in eval_pair_list]))
        res_df = self.summary(est_dict, **self.PerformanceEvaluation_params['summary_kwargs'])
        logger.info(res_df)
        return res_df

    def summary(self, est_dict, **kwargs):
        train_col_name = 'train_return_'
        if self.TrainRewardModel4Eval_params['mode'] == 'covariate_shift' and self.PerformanceEvaluation_params['covariate_shift']:
            train_col_name += '+'
        else:
            train_col_name += '-'

        if self.PerformanceEvaluation_params['doubly_robust']['importance_sampling']:
            train_col_name += '+'
        else:
            train_col_name += '-'

        res_df = pd.Series(index=[train_col_name+'+',
                                  train_col_name+'-',
                                  'test_return_with_mis_bias',
                                  'ground_truth_return',
                                  'estimated_opt_bias'])
        if self.OfflineTrainDataPreprocessing_params['eval_method'] == 'ground_truth':
            res_df['ground_truth_return'] = est_dict[('ground_truth', 'ground_truth')]
        elif self.OfflineTrainDataPreprocessing_params['eval_method'] == 'bootstrap':
            threshold = kwargs.get('threshold', 1.0)
            res_df['ground_truth_return'] = est_dict[('train', 'ground_truth')]
            res_df[train_col_name+'-'] = est_dict[('train', 'train')]
            res_df['test_return_with_mis_bias'] = est_dict[('train', 'test')]
            opt_bias_tensor = torch.tensor([est_dict[
                ('resampled_train_{}'.format(each_idx),
                 'resampled_train_{}'.format(each_idx))] \
                                            for each_idx in range(self.OfflineTrainDataPreprocessing_params['n_eval'])])
            opt_bias_tensor = opt_bias_tensor \
                - torch.tensor([est_dict[('resampled_train_{}'.format(each_idx),
                                          'train')] \
                                for each_idx in range(self.OfflineTrainDataPreprocessing_params['n_eval'])])
            mean_opt_bias = float(torch.mean(opt_bias_tensor))
            std_opt_bias = float(torch.std(opt_bias_tensor))
            logger.info(' * optimistic bias = {} +/- {}'.format(mean_opt_bias, std_opt_bias))
            if abs(mean_opt_bias) < threshold * std_opt_bias and kwargs.get('apply_decision_rule', False):
                logger.info('no bias reduction')
                mean_opt_bias = 0.0
            res_df['estimated_opt_bias'] = mean_opt_bias
            res_df[train_col_name+'+'] = res_df[train_col_name+'-'] - mean_opt_bias
        elif self.OfflineTrainDataPreprocessing_params['eval_method'] == 'train_test_split':
            res_df['ground_truth_return'] = est_dict[('train', 'ground_truth')]
            res_df[train_col_name+'-'] = est_dict[('train', 'train')]
            res_df['test_return_with_mis_bias'] = est_dict[('train', 'test')]
            res_df['estimated_opt_bias'] \
                = float(torch.mean(torch.tensor([est_dict[each_key_pair] for each_key_pair in est_dict.keys()
                                                 if each_key_pair[0].startswith('split_train')
                                                 and each_key_pair[1].startswith('split_train')])))\
                                                 - float(torch.mean(torch.tensor([est_dict[each_key_pair] for each_key_pair in est_dict.keys()
                                                                                  if each_key_pair[0].startswith('split_train')
                                                                                  and each_key_pair[1].startswith('split_test')])))
            res_df[train_col_name+'+'] = res_df[train_col_name+'-'] - res_df['estimated_opt_bias']
        return res_df



@requires(EnvTrainTestSetup, RunRandomTest, TrainRandomRewardModel4Eval, EvalPairList)
class RandomPerformanceEvaluation(PerformanceEvaluation):
    pass


class MultiplePerformanceEvaluation(AutoNamingTask, RngMixin):

    ReactantSelector_params = luigi.DictParameter()
    EnvSetup_params = luigi.DictParameter()
    OfflineTrainDataPreprocessing_params = luigi.DictParameter()
    OfflineTestDataPreprocessing_params = luigi.DictParameter()
    OfflineTrainDataConstruction_params = luigi.DictParameter()
    OfflineTestDataConstruction_params = luigi.DictParameter()
    EnvTrainTestSetup_params = luigi.DictParameter()
    FeatureSetup_params = luigi.DictParameter()
    AgentSetup_params = luigi.DictParameter()
    Train_params = luigi.DictParameter()
    TrainRewardModel4Eval_params = luigi.DictParameter()
    #TrainDensityRatio_params = luigi.DictParameter()
    RunTest_params = luigi.DictParameter()
    PerformanceEvaluation_params = luigi.DictParameter()
    seed = luigi.IntParameter()
    n_iter = luigi.IntParameter()

    def requires(self):
        self.set_seed(self.seed)
        seeded_dict = {}
        #seeded_dict['EnvSetup_params'] = self.EnvSetup_params
        seeded_dict['OfflineTrainDataConstruction_params'] = self.OfflineTrainDataConstruction_params
        seeded_dict['OfflineTestDataConstruction_params'] = self.OfflineTestDataConstruction_params
        #seeded_dict['OfflineTrainDataPreprocessing_params'] = self.OfflineTrainDataPreprocessing_params
        #seeded_dict['EnvTrainTestSetup_params'] = self.EnvTrainTestSetup_params
        #seeded_dict['AgentSetup_params'] = self.AgentSetup_params
        #seeded_dict['Train_params'] = self.Train_params
        #seeded_dict['RunTest_params'] = self.RunTest_params
        return [PerformanceEvaluation(
            ReactantSelector_params=self.ReactantSelector_params,
            EnvSetup_params=self.EnvSetup_params, #each_seeded_dict['EnvSetup_params'],
            OfflineTrainDataConstruction_params=each_seeded_dict['OfflineTrainDataConstruction_params'],
            OfflineTestDataConstruction_params=each_seeded_dict['OfflineTestDataConstruction_params'],
            OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
            OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
            EnvTrainTestSetup_params=self.EnvTrainTestSetup_params,
            FeatureSetup_params=self.FeatureSetup_params,
            AgentSetup_params=self.AgentSetup_params,
            Train_params=self.Train_params,
            TrainRewardModel4Eval_params=self.TrainRewardModel4Eval_params,
            RunTest_params=self.RunTest_params,
            PerformanceEvaluation_params=self.PerformanceEvaluation_params,)
                for each_seeded_dict in self.seeding_iter(seeded_dict, self.n_iter)]

    def run_task(self, input_list):
        train_col_name = 'train_return_'

        if self.TrainRewardModel4Eval_params['mode'] == 'covariate_shift' and self.PerformanceEvaluation_params['covariate_shift']:
            train_col_name += '+'
        else:
            train_col_name += '-'

        if self.PerformanceEvaluation_params['doubly_robust']['importance_sampling']:
            train_col_name += '+'
        else:
            train_col_name += '-'

        res_df = pd.concat(input_list, axis=1).transpose()
        logger.info(res_df)
        mean_res_df = res_df.mean()
        std_res_df = res_df.std()
        output_dict = {
            'mean_{}-'.format(train_col_name): mean_res_df[train_col_name+'-'],
            'mean_test_return_with_mis_bias': mean_res_df['test_return_with_mis_bias'],
            'mean_ground_truth_return': mean_res_df['ground_truth_return'],
            'mean_opt_bias': mean_res_df[train_col_name+'-'] - mean_res_df['test_return_with_mis_bias'],
            'mean_mis_bias': mean_res_df['test_return_with_mis_bias'] - mean_res_df['ground_truth_return'],
            'mean_est_opt_bias': mean_res_df['estimated_opt_bias'],
            'mean_{}+'.format(train_col_name): mean_res_df[train_col_name+'+'],
            'std_{}-'.format(train_col_name): std_res_df[train_col_name+'-'],
            'std_test_return_with_mis_bias': std_res_df['test_return_with_mis_bias'],
            'std_ground_truth_return': std_res_df['ground_truth_return'],
            'std_opt_bias': (res_df[train_col_name+'-'] - res_df['test_return_with_mis_bias']).std(),
            'std_mis_bias': (res_df['test_return_with_mis_bias'] - res_df['ground_truth_return']).std(),
            'std_est_opt_bias': std_res_df['estimated_opt_bias'],
            'std_{}+'.format(train_col_name): res_df[train_col_name+'+'].std()}
        logger.info('\n'+str(pd.Series(output_dict)))
        return output_dict


class MultipleRandomPerformanceEvaluation(MultiplePerformanceEvaluation):

    ReactantSelector_params = luigi.DictParameter()
    EnvSetup_params = luigi.DictParameter()
    OfflineTrainDataPreprocessing_params = luigi.DictParameter()
    OfflineTestDataPreprocessing_params = luigi.DictParameter()
    OfflineTrainDataConstruction_params = luigi.DictParameter()
    OfflineTestDataConstruction_params = luigi.DictParameter()
    EnvTrainTestSetup_params = luigi.DictParameter()
    FeatureSetup_params = luigi.DictParameter()
    AgentSetup_params = luigi.DictParameter(default=dict())
    Train_params = luigi.DictParameter()
    TrainRewardModel4Eval_params = luigi.DictParameter()
    RunTest_params = luigi.DictParameter()
    PerformanceEvaluation_params = luigi.DictParameter()
    seed = luigi.IntParameter()
    n_iter = luigi.IntParameter()

    def requires(self):
        self.set_seed(self.seed)
        seeded_dict = {}
        #seeded_dict['EnvSetup_params'] = self.EnvSetup_params
        seeded_dict['OfflineTrainDataConstruction_params'] = self.OfflineTrainDataConstruction_params
        seeded_dict['OfflineTestDataConstruction_params'] = self.OfflineTestDataConstruction_params
        #seeded_dict['OfflineDataPreprocessing_params'] = self.OfflineDataPreprocessing_params
        #seeded_dict['EnvTrainTestSetup_params'] = self.EnvTrainTestSetup_params
        #seeded_dict['AgentSetup_params'] = self.AgentSetup_params
        #seeded_dict['Train_params'] = self.Train_params
        #seeded_dict['RunTest_params'] = self.RunTest_params
        return [RandomPerformanceEvaluation(
            ReactantSelector_params=self.ReactantSelector_params,
            EnvSetup_params=self.EnvSetup_params, #each_seeded_dict['EnvSetup_params'],
            OfflineTrainDataConstruction_params=each_seeded_dict['OfflineTrainDataConstruction_params'],
            OfflineTestDataConstruction_params=each_seeded_dict['OfflineTestDataConstruction_params'],
            OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
            OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
            EnvTrainTestSetup_params=self.EnvTrainTestSetup_params,
            FeatureSetup_params=self.FeatureSetup_params,
            AgentSetup_params=self.AgentSetup_params,
            Train_params=self.Train_params,
            TrainRewardModel4Eval_params=self.TrainRewardModel4Eval_params,
            RunTest_params=self.RunTest_params,
            PerformanceEvaluation_params=self.PerformanceEvaluation_params,)
                for each_seeded_dict in self.seeding_iter(seeded_dict, self.n_iter)]


class MultipleRun(MultipleRunBase):

    output_ext = luigi.Parameter(default='pkl')

    def obj_task(self, **kwargs):
        return MultiplePerformanceEvaluation(**kwargs)


class PlotBiasEstimation(LinePlotMultipleRun):

    def requires(self):
        return [MultipleRun(MultipleRun_params=self.MultipleRun_params)]



class EvaluateComputationTime(AutoNamingTask, RngMixin):

    seed = luigi.IntParameter()
    n_iter = luigi.IntParameter()

    def requires(self):
        return [EnvTrainTestSetup(
            ReactantSelector_params=self.ReactantSelector_params,
            EnvSetup_params=self.EnvSetup_params,
            OfflineTrainDataConstruction_params=self.OfflineTrainDataConstruction_params,
            OfflineTestDataConstruction_params=self.OfflineTestDataConstruction_params,
            OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
            OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
            EnvTrainTestSetup_params=self.EnvTrainTestSetup_params)]

    def run_task(self, input_list):
        
        self.set_seed(self.seed)
        seeded_dict = {}
        #seeded_dict['EnvSetup_params'] = self.EnvSetup_params
        #seeded_dict['OfflineDataConstruction_params'] = self.OfflineDataConstruction_params
        #seeded_dict['OfflineDataPreprocessing_params'] = self.OfflineDataPreprocessing_params
        seeded_dict['FeatureSetup_params'] = dict(self.FeatureSetup_params)
        seeded_dict['FeatureSetup_params']['seed'] = 0
        #seeded_dict['EnvTrainTestSetup_params'] = self.EnvTrainTestSetup_params
        #seeded_dict['AgentSetup_params'] = self.AgentSetup_params
        #seeded_dict['Train_params'] = self.Train_params
        #seeded_dict['RunTest_params'] = self.RunTest_params
        from time import time
        feature_construction_time_list = []
        train_time_list = []
        for each_seeded_dict in self.seeding_iter(seeded_dict, self.n_iter):
            logger.info(' * start feature construction *')
            task = FeatureSetup(
                ReactantSelector_params=self.ReactantSelector_params,
                EnvSetup_params=self.EnvSetup_params, 
                OfflineTrainDataConstruction_params=self.OfflineTrainDataConstruction_params,
                OfflineTestDataConstruction_params=self.OfflineTestDataConstruction_params,
                OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
                OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
                EnvTrainTestSetup_params=self.EnvTrainTestSetup_params,
                FeatureSetup_params=each_seeded_dict['FeatureSetup_params'])
            pathlib.Path(self._working_dir / 'OUTPUT' / task.working_subdir).mkdir(exist_ok=True)
            try:
                shutil.rmtree(self._working_dir / 'OUTPUT' / task.working_subdir / 'zdd')
            except:
                pass
            s1 = time()
            task.run()
            feature_construction_time = time() - s1
            feature_construction_time_list.append(feature_construction_time)
            logger.info(' * end feature construction *')

            task = AgentSetup(
                ReactantSelector_params=self.ReactantSelector_params,
                EnvSetup_params=self.EnvSetup_params, #each_seeded_dict['EnvSetup_params'],
                OfflineTrainDataConstruction_params=self.OfflineTrainDataConstruction_params,
                OfflineTestDataConstruction_params=self.OfflineTestDataConstruction_params,
                OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
                OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
                EnvTrainTestSetup_params=self.EnvTrainTestSetup_params,
                FeatureSetup_params=each_seeded_dict['FeatureSetup_params'],
                AgentSetup_params=self.AgentSetup_params)
            pathlib.Path(self._working_dir / 'OUTPUT' / task.working_subdir).mkdir(exist_ok=True)
            task.run()


            logger.info(' * start train *')
            task = Train(
                ReactantSelector_params=self.ReactantSelector_params,
                EnvSetup_params=self.EnvSetup_params, #each_seeded_dict['EnvSetup_params'],
                OfflineTrainDataConstruction_params=self.OfflineTrainDataConstruction_params,
                OfflineTestDataConstruction_params=self.OfflineTestDataConstruction_params,
                OfflineTrainDataPreprocessing_params=self.OfflineTrainDataPreprocessing_params,
                OfflineTestDataPreprocessing_params=self.OfflineTestDataPreprocessing_params,
                EnvTrainTestSetup_params=self.EnvTrainTestSetup_params,
                FeatureSetup_params=each_seeded_dict['FeatureSetup_params'],
                AgentSetup_params=self.AgentSetup_params,
                Train_params=self.Train_params)
            pathlib.Path(self._working_dir / 'OUTPUT' / task.working_subdir).mkdir(exist_ok=True)
            s1 = time()
            task.run()
            train_time = time() - s1
            train_time_list.append(train_time)
            logger.info(' * end train *')
        logger.info(' ** computation time ** ')
        logger.info(' * feature construction:\t\t\t {} +/- {} sec'.format(np.mean(feature_construction_time_list),
                                                                          np.std(feature_construction_time_list)))
        logger.info(' * train:\t\t\t\t {} +/- {} sec'.format(np.mean(train_time_list),
                                                             np.std(train_time_list)))
        return feature_construction_time_list, train_time_list
