from copy import deepcopy
import logging
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import luigi
from luigi.util import requires
import numpy as np
from rdkit import Chem
from luigine.abc import AutoNamingTask
import torch

import covshift.weight
from molgym.chem.density_ratio import FingerprintWeightEstimator
from molgym.rl.episode import EpisodeMemory
import molgym.chem.fingerprint
import molgym.chem.qsar
import molgym.rl.agent
import molgym.rl.feature
import molgym.rl.valfunc
from mlbasics.utils import RngMixin, DeviceContext, device_count, device_name
from .env import OfflineTrainDataConstruction, OfflineTrainDataPreprocessing, OfflineTestDataPreprocessing, EvalPairList, EnvTrainTestSetup

logger = logging.getLogger('luigi-interface')
AutoNamingTask.output_ext = luigi.Parameter(default='pkl')


@requires(EnvTrainTestSetup, OfflineTrainDataConstruction)
class FeatureSetup(AutoNamingTask):

    ''' Pre-compute ZDD using the offline data set.
    '''

    FeatureSetup_params = luigi.DictParameter()

    def run_task(self, input_list):
        env, _, _ = input_list[0]
        train_episode_memory = input_list[1]

        fingerprint = getattr(molgym.chem.fingerprint,
                              self.FeatureSetup_params['fingerprint']['module'])(
                                  **self.FeatureSetup_params['fingerprint']['kwargs'])
        if hasattr(fingerprint, 'prep'):
            fingerprint.prep(train_episode_memory=train_episode_memory, 
                             workers=self.FeatureSetup_params['workers'],
                             per_batch_n_jobs=self.FeatureSetup_params['per_batch_n_jobs'],
                             dir_obj=(self._working_dir / 'OUTPUT' / self.working_subdir / 'zdd'),
                             action_space=env.action_space,
                             logger=logger.info)

        state_action_feature = getattr(
            molgym.rl.feature,
            self.FeatureSetup_params['state_action_feature']['module'])(
                fingerprint=fingerprint,
                action_space=env.action_space,
                **self.FeatureSetup_params['state_action_feature']['kwargs'])
        return state_action_feature


@requires(EnvTrainTestSetup, FeatureSetup, EvalPairList)
class AgentSetup(AutoNamingTask, RngMixin):

    AgentSetup_params = luigi.DictParameter()

    def run_task(self, input_list):
        env_train, _, _ = input_list[0]
        state_action_feature = input_list[1]
        eval_pair_list = input_list[2]

        agent_dict = {}
        self.set_seed(self.AgentSetup_params['agent']['seed'])
        seed = self.gen_seed()

        # collect train_keys
        train_key_list = sorted(list(set(list(zip(*eval_pair_list))[0])))

        for each_key in train_key_list:
            if 'Greedy' not in self.AgentSetup_params['agent']['module']:
                action_val_model = getattr(molgym.rl.valfunc,
                                           self.AgentSetup_params['action_val_model']['module'])(
                                               state_action_feature=state_action_feature,
                                               **self.AgentSetup_params['action_val_model']['kwargs'])
                agent = getattr(molgym.rl.agent, self.AgentSetup_params['agent']['module'])(
                    action_space=env_train.action_space,
                    seed=seed,
                    action_val_model=action_val_model,
                    max_step=env_train.max_step,
                    **self.AgentSetup_params['agent']['kwargs'])
            else:
                agent = getattr(molgym.rl.agent,
                                self.AgentSetup_params['agent']['module'])(
                                    action_space=env_train.action_space,
                                    qsar_model=env_train.qsar_model,
                                    max_step=env_train.max_step)
            agent_dict[each_key] = agent
            if not self.AgentSetup_params['use_same_seed']:
                seed = self.gen_seed()
        return agent_dict


def _train(device_idx, train_key, property_tuple, env_train, agent, env_seed, agent_seed,
           working_dir, Train_params, AgentSetup_params):
    logging.basicConfig(
        level=logging.INFO,
        filename=str(working_dir / 'ENGLOG' / 'engine.log'))
    logger = logging.getLogger('luigi-interface.task-{}'.format(train_key))
    env_train.qsar_model = property_tuple.qsar_model
    env_train.set_seed(env_seed)
    agent.set_seed(agent_seed)
    env_train.to(device_name(device_idx))
    agent.to(device_name(device_idx))
    logger.info('to device: {}'.format(device_name(device_idx)))
    with DeviceContext(device_idx):
        agent.fit(property_tuple.episode_memory,
                  env_train,
                  logger=logger.info,
                  **Train_params['fit_kwargs'])
    if hasattr(agent, 'delete_rng'):
        agent.delete_rng()
    agent.to('cpu')
    return (train_key, agent)


@requires(EnvTrainTestSetup, AgentSetup, EvalPairList)
class Train(AutoNamingTask, RngMixin):

    Train_params = luigi.DictParameter()

    def run_task(self, input_list):
        env_train, property_tuple_dict, _ = input_list[0]
        agent_dict = input_list[1]
        eval_pair_list = input_list[2]
        train_key_list = sorted(list(set(list(zip(*eval_pair_list))[0])))

        self.set_seed(self.Train_params['seed'])
        if self.Train_params['use_same_seed']:
            agent_seed = self.gen_seed()
            env_seed = self.gen_seed()
            agent_seed_list = [agent_seed] * len(train_key_list)
            env_seed_list = [env_seed] * len(train_key_list)
        else:
            agent_seed_list = self.seed_list(len(train_key_list))
            env_seed_list = self.seed_list(len(train_key_list))

        if self.Train_params['workers'] == 1:
            agent_dict = dict([_train(each_idx % device_count(),
                                      each_key,
                                      property_tuple_dict[each_key],
                                      env_train,
                                      agent_dict[each_key],
                                      env_seed=env_seed_list[each_idx],
                                      agent_seed=agent_seed_list[each_idx],
                                      working_dir=self._working_dir,
                                      Train_params=self.Train_params,
                                      AgentSetup_params=self.AgentSetup_params)
                               for each_idx, each_key in enumerate(train_key_list)])
        else:
            agent_dict = dict(Parallel(n_jobs=self.Train_params['workers'],
                                       verbose=10)([
                                           delayed(_train)(
                                               each_idx % device_count(),
                                               each_key,
                                               property_tuple_dict[each_key],
                                               env_train,
                                               agent_dict[each_key],
                                               env_seed=env_seed_list[each_idx],
                                               agent_seed=agent_seed_list[each_idx],
                                               working_dir=self._working_dir,
                                               Train_params=self.Train_params,
                                               AgentSetup_params=self.AgentSetup_params)
                                           for each_idx, each_key in enumerate(train_key_list)]))
            get_reusable_executor().shutdown(wait=True)
        return agent_dict


def _gen_episode(device_idx, agent_idx, agent, agent_seed_list, env_seed_list,
                 start_mol_list, train_env):
    n_episode = len(start_mol_list)
    episode_list = []
    agent.to(device_name(device_idx))
    train_env.to(device_name(device_idx))
    with DeviceContext(device_idx):
        for each_episode_idx in range(n_episode):
            agent.set_seed(agent_seed_list[each_episode_idx])
            train_env.set_seed(env_seed_list[each_episode_idx])
            episode_list.append(agent.gen_episode(train_env, current_mol=start_mol_list[each_episode_idx]))
    return agent_idx, EpisodeMemory(episode_list, seed=0)


@requires(EnvTrainTestSetup, Train)
class RunTest(AutoNamingTask, RngMixin):

    RunTest_params = luigi.DictParameter()

    def run_task(self, input_list):
        train_env, _, _ = input_list[0]
        agent_dict = input_list[1]
        self.set_seed(self.RunTest_params['seed'])

        start_mol_list = train_env.action_space.get_reactant_list(self.RunTest_params['n_trial'])
        train_env.to(self.RunTest_params['device'])
        train_env.qsar_model.forward(Chem.MolFromSmiles('c1ccccc1'))
        train_env.qsar_model.forward(Chem.MolFromSmiles('c1ccccc1'))

        # preparation
        n_agent = len(agent_dict)
        n_total_tasks = (n_agent * self.RunTest_params['n_trial'])
        agent_seed_list = self.seed_list(self.RunTest_params['n_trial'])
        env_seed_list = self.seed_list(self.RunTest_params['n_trial'])

        # run each resampled agent on the test env
        if len(agent_dict) != 1:
            episode_list_dict = dict(Parallel(n_jobs=self.RunTest_params['workers'], verbose=10)([
                    delayed(_gen_episode)(
                        each_device_idx % device_count() if self.RunTest_params['device'] != 'cpu' else 'cpu',
                        each_agent_idx,
                        each_agent,
                        agent_seed_list=agent_seed_list,
                        env_seed_list=env_seed_list,
                        start_mol_list=start_mol_list,
                        train_env=train_env)
                for each_device_idx, (each_agent_idx, each_agent) in enumerate(agent_dict.items())
            ]))
        else:
            n_task = int(np.ceil(self.RunTest_params['n_trial'] / self.RunTest_params['workers']))
            agent_key = list(agent_dict.keys())[0]
            agent = agent_dict[agent_key]
            episode_memory_list = list(list(zip(*Parallel(n_jobs=self.RunTest_params['workers'], verbose=10)([
                delayed(_gen_episode)(
                    device_idx=0,
                    agent_idx=0,
                    agent=agent,
                    agent_seed_list=agent_seed_list[n_task * each_work : n_task * (each_work + 1)],
                    env_seed_list=env_seed_list[n_task * each_work : n_task * (each_work + 1)],
                    start_mol_list=start_mol_list[n_task * each_work : n_task * (each_work + 1)],
                    train_env=train_env
                )
                for each_work in range(self.RunTest_params['workers'])
            ])))[1])
            episode_memory = episode_memory_list.pop(0)
            for each_episode_memory in episode_memory_list:
                episode_memory = episode_memory + each_episode_memory
            episode_list_dict = {agent_key: episode_memory}
        get_reusable_executor().shutdown(wait=True)
        return episode_list_dict


@requires(EnvTrainTestSetup, EvalPairList)
class RunRandomTest(RunTest):

    def run_task(self, input_list):
        train_env, _, _ = input_list[0]
        eval_pair_list = input_list[1]

        train_key_list = sorted(list(set(list(zip(*eval_pair_list))[0])))
        behavior_agent = getattr(
            molgym.rl.agent,
            self.OfflineTrainDataConstruction_params['behavior_agent']['name'])(
                seed=self.OfflineTrainDataConstruction_params['behavior_agent']['seed'],
                working_dir=self._working_dir,
                max_step=train_env.max_step,
                **self.OfflineTrainDataConstruction_params['behavior_agent']['kwargs'])
        agent_dict = {each_key: deepcopy(behavior_agent) for each_key in train_key_list}
        
        self.set_seed(self.RunTest_params['seed'])

        start_mol_list = train_env.action_space.get_reactant_list(self.RunTest_params['n_trial'])
        train_env.qsar_model.forward(Chem.MolFromSmiles('c1ccccc1'))
        train_env.qsar_model.forward(Chem.MolFromSmiles('c1ccccc1'))

        # preparation
        n_agent = len(agent_dict)
        n_total_tasks = (n_agent * self.RunTest_params['n_trial'])
        agent_seed_list = self.seed_list(self.RunTest_params['n_trial'])
        env_seed_list = self.seed_list(self.RunTest_params['n_trial'])

        # run each resampled agent on the test env
        episode_list_dict = dict(Parallel(n_jobs=self.RunTest_params['workers'], verbose=10)([
                delayed(_gen_episode)(
                    each_device_idx % device_count(),
                    each_agent_idx,
                    each_agent,
                    agent_seed_list=agent_seed_list,
                    env_seed_list=env_seed_list,
                    start_mol_list=start_mol_list,
                    train_env=train_env)
            for each_device_idx, (each_agent_idx, each_agent) in enumerate(agent_dict.items())
        ]))
        get_reusable_executor().shutdown(wait=True)
        return episode_list_dict


def _train_reward(device_idx,
                  task_key,
                  src_episode_memory,
                  tgt_episode_memory,
                  qsar_model,
                  seed,
                  working_dir,
                  EnvTrainTestSetup_params,
                  TrainRewardModel4Eval_params):
    '''
    src_episode_memory: train set
    tgt_episode_memory: test set sampled in RunTest
    '''
    logging.basicConfig(
        level=logging.INFO,
        filename=str(working_dir / 'ENGLOG' / 'engine.log'))
    logger = logging.getLogger('luigi-interface.task-{}'.format(task_key))
    with DeviceContext(device_idx):
        density_ratio_estimator = FingerprintWeightEstimator(
            TrainRewardModel4Eval_params['density_ratio']['fingerprint'],
            TrainRewardModel4Eval_params['density_ratio']['covshift'])

        src_mol_list, src_tgt_list = src_episode_memory.extract_dataset()
        tgt_mol_list, _ = tgt_episode_memory.extract_dataset()

        density_ratio = torch.Tensor(density_ratio_estimator.fit(
            src_mol_list, tgt_mol_list,
            src_tgt_list=src_tgt_list,
            logger=logger.info,
            **TrainRewardModel4Eval_params['density_ratio']['fingerprint'].get('fit_kwargs', {}))).to(device_idx)

        if TrainRewardModel4Eval_params['mode'] == 'covariate_shift':
            fingerprint = getattr(molgym.chem.fingerprint,
                                  EnvTrainTestSetup_params['fingerprint']['module'])(
                                      **EnvTrainTestSetup_params['fingerprint']['kwargs'])
            qsar_model = getattr(molgym.chem.qsar,
                                 EnvTrainTestSetup_params['qsar_model']['module'])(
                                     fingerprint=fingerprint,
                                     seed=seed,
                                     **EnvTrainTestSetup_params['qsar_model']['kwargs'])
            qsar_model.fit(src_mol_list,
                           src_tgt_list,
                           weight=density_ratio,
                           weight_decay=TrainRewardModel4Eval_params['weight_decay'],
                           logger=logger.info,
                           **EnvTrainTestSetup_params['fit_kwargs'])
    qsar_model.delete_rng()
    qsar_model.to('cpu')
    return task_key, (qsar_model, density_ratio)


def _train_density_ratio(device_idx,
                         task_key,
                         n_step,
                         src_episode_memory,
                         tgt_episode_memory,
                         working_dir,
                         TrainDensityRatio_params):
    '''
    src_episode_memory: train set
    tgt_episode_memory: test set sampled in RunTest
    '''
    logging.basicConfig(
        level=logging.INFO,
        filename=str(working_dir / 'ENGLOG' / 'engine.log'))
    logger = logging.getLogger('luigi-interface.task-{}'.format(task_key))
    with DeviceContext(device_idx):
        density_ratio_estimator = FingerprintWeightEstimator(
            TrainDensityRatio_params['fingerprint'],
            TrainDensityRatio_params['covshift'])
        src_mol_list = src_episode_memory.done_mol_list
        tgt_mol_list = tgt_episode_memory.get_mol_list(n_step)
        density_ratio = torch.Tensor(density_ratio_estimator.fit(src_mol_list, tgt_mol_list))
    return task_key, density_ratio


@requires(OfflineTrainDataPreprocessing, OfflineTestDataPreprocessing, EnvTrainTestSetup, RunTest, EvalPairList)
class TrainRewardModel4Eval(AutoNamingTask, RngMixin):

    TrainRewardModel4Eval_params = luigi.DictParameter()

    def run_task(self, input_list):
        train_eval_pair_list, train_episode_memory_dict = input_list[0]
        test_eval_pair_list, test_episode_memory_dict = input_list[1]
        _, property_tuple_dict, _ = input_list[2]
        episode_list_dict = input_list[3]
        eval_pair_list = input_list[4]
        if self.TrainRewardModel4Eval_params['mode'] == 'skip':
            qsar_model_dict = {each_key: (property_tuple_dict[each_key[1]].qsar_model, None) for each_key in eval_pair_list}
            return qsar_model_dict

        train_episode_memory_dict.update(test_episode_memory_dict)
        
        # set seed
        self.set_seed(self.TrainRewardModel4Eval_params['seed'])
        seed = self.gen_seed()
        if self.EnvTrainTestSetup_params['use_same_seed']:
            seed_list = [seed] * len(eval_pair_list)
        else:
            seed_list = self.seed_list(len(eval_pair_list))

        # set ground_truth
        qsar_model_dict = {}
        try:
            eval_pair_list.remove(('train', 'ground_truth'))
            qsar_model_dict[('train', 'ground_truth')] = (property_tuple_dict['ground_truth'].qsar_model,
                                                          None)
        except ValueError:
            pass
        try:
            eval_pair_list.remove(('train', 'test'))
            qsar_model_dict[('train', 'test')] = (property_tuple_dict['test'].qsar_model,
                                                  None)
        except ValueError:
            pass
        qsar_model_dict.update(dict(Parallel(n_jobs=self.TrainRewardModel4Eval_params['workers'],
                                             verbose=10)([
                                                 delayed(_train_reward)(
                                                     each_idx % device_count(),
                                                     (each_train_key, each_test_key),
                                                     train_episode_memory_dict[each_test_key],
                                                     episode_list_dict[each_train_key],
                                                     property_tuple_dict[each_test_key].qsar_model,
                                                     seed_list[each_idx],
                                                     self._working_dir,
                                                     self.EnvTrainTestSetup_params,
                                                     self.TrainRewardModel4Eval_params)\
                                                 for each_idx, (each_train_key, each_test_key) \
                                                 in enumerate(eval_pair_list)])))
        return qsar_model_dict


@requires(OfflineTrainDataPreprocessing, OfflineTestDataPreprocessing, EnvTrainTestSetup, RunRandomTest, EvalPairList)
class TrainRandomRewardModel4Eval(TrainRewardModel4Eval):
    pass



@requires(OfflineTrainDataPreprocessing, EnvTrainTestSetup, RunTest)
class TrainDensityRatio(AutoNamingTask):

    TrainDensityRatio_params = luigi.DictParameter()

    def run_task(self, input_list):
        if self.TrainDensityRatio_params['skip']:
            return {}
        train_eval_pair_list, train_episode_memory_dict = input_list[0]
        train_env, property_tuple_dict, _ = input_list[1]
        episode_list_dict = input_list[2]

        density_ratio_dict_list = []
        each_step = train_env.max_step
        
        #for each_step in range(train_env.max_step+1):
        density_ratio_dict = {}
        density_ratio_dict.update(dict(Parallel(
            n_jobs=self.TrainDensityRatio_params['workers'],
            verbose=10)([
                delayed(_train_density_ratio)(
                    each_idx % device_count(),
                    (each_train_key, each_test_key),
                    each_step,
                    train_episode_memory_dict[each_test_key],
                    episode_list_dict[each_train_key],
                    self._working_dir,
                    self.TrainDensityRatio_params)\
                for each_idx, (each_train_key, each_test_key) \
                in enumerate(train_eval_pair_list)])))
        return density_ratio_dict

@requires(OfflineTrainDataPreprocessing, EnvTrainTestSetup, RunRandomTest)
class TrainRandomDensityRatio(TrainDensityRatio):
    pass
