#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple, Counter
import logging
from joblib import Parallel, delayed
import luigi
from luigi.util import requires
import numpy as np
import os
import torch

from luigine.abc import AutoNamingTask
import molgym.chem.qsar
import molgym.chem.fingerprint
import molgym.rl.agent
import molgym.rl.env
from molgym.utils import construct_dataset
from mlbasics.utils import RngMixin, DeviceContext, device_count
from molgym.rl.episode import episode_memory_from_offline_data

logger = logging.getLogger('luigi-interface')
AutoNamingTask.output_ext = luigi.Parameter(default='pkl')

# Data
PropertyTuple = namedtuple('PropertyTuple', ['qsar_model',
                                             'episode_memory'])


class GrammarInference(AutoNamingTask):
    pass


class ReactantSelector(AutoNamingTask):

    ReactantSelector_params = luigi.DictParameter()

    def run(self):
        if self.ReactantSelector_params['n_mol'] == -1:
            os.system('cp {} {}'.format(str(self._working_dir / self.ReactantSelector_params['input_path']),
                                        self.output().path))
        else:
            os.system('/bin/bash -c "shuf {} --output={} --random-source=<(openssl enc -aes-256-ctr -pass pass:{} -nosalt  </dev/zero 2>/dev/null) -n {}"'.format(
                str(self._working_dir / self.ReactantSelector_params['input_path']),
                self.output().path,
                self.ReactantSelector_params['seed'],
                self.ReactantSelector_params['n_mol']))

    def load_output(self):
        return self.output().path


@requires(ReactantSelector)
class EnvSetup(AutoNamingTask):

    ''' set up the test environment w/ the original QSAR model
    '''

    EnvSetup_params = luigi.DictParameter()

    def run_task(self, input_list):
        qsar_model = getattr(molgym.chem.qsar,
                             self.EnvSetup_params['qsar_model']['module'])(
                                 **self.EnvSetup_params['qsar_model']['kwargs'])
        env = getattr(molgym.rl.env, self.EnvSetup_params['env']['module'])(
            seed=self.EnvSetup_params['env']['seed'],
            reactant_path=input_list[0],
            reaction_tsv_path=self.input_file()[0],
            qsar_model=qsar_model,
            **self.EnvSetup_params['env']['kwargs'])
        logger.info(' * # of reactions = {}'.format(env.action_space.reaction_corpus.n_reaction))
        logger.info(' * # of reactants = {}'.format(env.action_space.reaction_corpus.n_reactant))
        reactant_counter = Counter()
        for each_reaction in env.action_space.reaction_corpus.reaction_list:
            reactant_counter[each_reaction.n_reactant] += 1
        logger.info(' * freq of (# of reactants): {}'.format(reactant_counter))
        return env

    def input_file(self):
        return [str(self._working_dir
                    / self.EnvSetup_params['data']['reaction_tsv_path'])]


@requires(EnvSetup)
class OfflineTrainDataConstruction(AutoNamingTask, RngMixin):

    ''' Construct an offline data set by running a behavior policy on the test env.
    The offline data set may contain a pseudo episode generated from the mol-tgt data set,
    if `include_offline_data` is `True`.
    '''

    OfflineTrainDataConstruction_params = luigi.DictParameter()

    def run_task(self, input_list):
        env = input_list[0]
        self.set_seed(self.OfflineTrainDataConstruction_params['seed'])
        behavior_agent = getattr(
            molgym.rl.agent,
            self.OfflineTrainDataConstruction_params['behavior_agent']['name'])(
                seed=self.OfflineTrainDataConstruction_params['behavior_agent']['seed'],
                working_dir=self._working_dir,
                max_step=env.max_step,
                **self.OfflineTrainDataConstruction_params['behavior_agent']['kwargs'])
        behavior_agent.set_seed(self.OfflineTrainDataConstruction_params['behavior_agent']['seed'])
        env.set_seed(self.gen_seed())
        behavior_agent.to('cpu')
        env.to('cpu')
        train_episode_memory = behavior_agent.gen_episode_memory(
            env,
            **self.OfflineTrainDataConstruction_params['gen_episode_memory_kwargs'])
        logger.info(' * #(transitions) = {}'.format(train_episode_memory.n_transition))
        return train_episode_memory


@requires(EnvSetup)
class OfflineTestDataConstruction(AutoNamingTask, RngMixin):

    ''' Construct an offline data set by running a behavior policy on the test env.
    The offline data set may contain a pseudo episode generated from the mol-tgt data set,
    if `include_offline_data` is `True`.
    '''

    OfflineTestDataConstruction_params = luigi.DictParameter()

    def run_task(self, input_list):
        env = input_list[0]
        self.set_seed(self.OfflineTestDataConstruction_params['seed'])
        behavior_agent = getattr(
            molgym.rl.agent,
            self.OfflineTestDataConstruction_params['behavior_agent']['name'])(
                seed=self.OfflineTestDataConstruction_params['behavior_agent']['seed'],
                working_dir=self._working_dir,
                max_step=env.max_step,
                **self.OfflineTestDataConstruction_params['behavior_agent']['kwargs'])
        behavior_agent.set_seed(self.OfflineTestDataConstruction_params['behavior_agent']['seed'])
        env.set_seed(self.gen_seed())
        behavior_agent.to('cpu')
        env.to('cpu')
        test_episode_memory = behavior_agent.gen_episode_memory(
            env,
            **self.OfflineTestDataConstruction_params['gen_test_episode_memory_kwargs'])
        logger.info(' * #(transitions) = {}'.format(test_episode_memory.n_transition))

        ground_truth_episode_memory = episode_memory_from_offline_data(
            env,
            self.gen_seed(),
            self._working_dir,
            self.OfflineTestDataConstruction_params['ground_truth']['csv_path'],
            self.OfflineTestDataConstruction_params['ground_truth']['mol_col'],
            self.OfflineTestDataConstruction_params['ground_truth']['tgt_col'],
            self.OfflineTestDataConstruction_params['ground_truth']['preprocessing_list'])
        return test_episode_memory, ground_truth_episode_memory


@requires(OfflineTrainDataConstruction)
class OfflineTrainDataPreprocessing(AutoNamingTask):

    ''' set up train-test datasets for train-test split or bootstrap.

    - episode_memory_dict contains all of the memory dictionaries
    - train_test_episode_memory_list contains (train_key, test_key) pairs
    '''
    OfflineTrainDataPreprocessing_params = luigi.DictParameter()

    def run_task(self, input_list):
        episode_memory = input_list[0]
        episode_memory.set_seed(self.OfflineTrainDataPreprocessing_params['seed'])
        train_episode_memory_dict = {'train': episode_memory}
        train_eval_pair_list = [('train', 'train')]
        if self.OfflineTrainDataPreprocessing_params['eval_method'] == 'train_test_split':
            for each_idx in range(self.OfflineTrainDataPreprocessing_params['n_eval']):
                train_episode_memory, test_episode_memory \
                    = episode_memory.train_test_random_split(
                        **self.OfflineTrainDataPreprocessing_params['eval_kwargs'])
                train_episode_memory_dict['split_train_{}'.format(each_idx)] = train_episode_memory
                train_episode_memory_dict['split_test_{}'.format(each_idx)] = test_episode_memory
                train_eval_pair_list.append(('split_train_{}'.format(each_idx),
                                             'split_train_{}'.format(each_idx)))
                train_eval_pair_list.append(('split_train_{}'.format(each_idx),
                                             'split_test_{}'.format(each_idx)))
        elif self.OfflineTrainDataPreprocessing_params['eval_method'] == 'bootstrap':
            for each_idx in range(self.OfflineTrainDataPreprocessing_params['n_eval']):
                resampled_episode_memory = episode_memory.resample_episode()
                train_episode_memory_dict['resampled_train_{}'.format(each_idx)] = resampled_episode_memory
                train_eval_pair_list.append(('resampled_train_{}'.format(each_idx),
                                             'resampled_train_{}'.format(each_idx)))
                train_eval_pair_list.append(('resampled_train_{}'.format(each_idx),
                                             'train'))
        elif self.OfflineTrainDataPreprocessing_params['eval_method'] == 'ground_truth':
            train_eval_pair_list = []
            train_episode_memory_dict = {'train': episode_memory}
        else:
            raise ValueError('{} is not supported.'.format(self.OfflineTrainDataPreprocessing_params['eval_method']))
        return train_eval_pair_list, train_episode_memory_dict


@requires(OfflineTestDataConstruction)
class OfflineTestDataPreprocessing(AutoNamingTask):

    ''' set up train-test datasets for train-test split or bootstrap.

    - test_episode_memory_dict contains all of the memory dictionaries
    - train_test_episode_memory_list contains (train_key, test_key) pairs
    '''
    OfflineTestDataPreprocessing_params = luigi.DictParameter()

    def run_task(self, input_list):
        large_episode_memory, ground_truth_episode_memory = input_list[0]
        test_episode_memory_dict = {'test': large_episode_memory,
                                    'ground_truth': ground_truth_episode_memory}
        test_eval_pair_list = [('train', 'test'), ('train', 'ground_truth')]
        if self.OfflineTestDataPreprocessing_params['eval_method'] == 'ground_truth':
            test_eval_pair_list = [('ground_truth', 'ground_truth')]
            test_episode_memory_dict = {'ground_truth': ground_truth_episode_memory}
        return test_eval_pair_list, test_episode_memory_dict


@requires(OfflineTrainDataPreprocessing, OfflineTestDataPreprocessing)
class EvalPairList(AutoNamingTask):

    def run_task(self, input_list):
        train_eval_pair_list, _ = input_list[0]
        test_eval_pair_list, _ = input_list[1]
        return train_eval_pair_list + test_eval_pair_list


def _resample(device_idx,
              task_key,
              resampled_episode_memory,
              seed,
              working_dir,
              EnvTrainTestSetup_params):
    logging.basicConfig(
        level=logging.INFO,
        filename=str(working_dir / 'ENGLOG' / 'engine.log'))
    logger = logging.getLogger('luigi-interface.task-{}'.format(task_key))
    with DeviceContext(device_idx):
        fingerprint = getattr(molgym.chem.fingerprint,
                              EnvTrainTestSetup_params['fingerprint']['module'])(
                                  **EnvTrainTestSetup_params['fingerprint']['kwargs'])
        resampled_qsar_model = getattr(molgym.chem.qsar,
                                       EnvTrainTestSetup_params['qsar_model']['module'])(
                                           fingerprint=fingerprint,
                                           seed=seed,
                                           **EnvTrainTestSetup_params['qsar_model']['kwargs'])
        resampled_mol_list, resampled_tgt_list = resampled_episode_memory.extract_dataset()
        logger.info(' * constructed data set')
        resampled_qsar_model.fit(resampled_mol_list,
                                 resampled_tgt_list,
                                 logger=logger.info,
                                 **EnvTrainTestSetup_params['fit_kwargs'])
    resampled_qsar_model.delete_rng()
    resampled_qsar_model.to('cpu')
    if task_key == 'test':
        resampled_episode_memory = None
    return task_key, PropertyTuple(qsar_model=resampled_qsar_model,
                                   episode_memory=resampled_episode_memory)

def _each_cv(device_idx,
             fold_idx,
             mol_list_train,
             tgt_list_train,
             mol_list_test,
             tgt_list_test,
             seed,
             working_dir,
             EnvTrainTestSetup_params):
    logging.basicConfig(
        level=logging.INFO,
        filename=str(working_dir / 'ENGLOG' / 'engine.log'))
    logger = logging.getLogger('luigi-interface.task-cv-{}'.format(fold_idx))
    import molgym.chem.fingerprint
    with DeviceContext(device_idx):
        fingerprint = getattr(molgym.chem.fingerprint,
                              EnvTrainTestSetup_params['fingerprint']['module'])(
                                  **EnvTrainTestSetup_params['fingerprint']['kwargs'])
        qsar_model = getattr(molgym.chem.qsar,
                             EnvTrainTestSetup_params['qsar_model']['module'])(
                                 fingerprint=fingerprint,
                                 seed=seed,
                                 **EnvTrainTestSetup_params['qsar_model']['kwargs'])
        logger.info(' * constructed data set')
        qsar_model.fit(mol_list_train,
                       tgt_list_train,
                       logger=logger.info,
                       **EnvTrainTestSetup_params['fit_kwargs'])
        with torch.no_grad():
            tgt_pred = qsar_model.batch_forward(mol_list_test).to('cpu')
    return torch.sqrt(torch.mean((torch.Tensor(tgt_list_test).ravel() - tgt_pred.ravel()) ** 2))


@requires(OfflineTrainDataPreprocessing, OfflineTestDataPreprocessing, EnvSetup)
class EnvTrainTestSetup(AutoNamingTask, RngMixin):

    ''' Construct training & test envs from the offline data set, in a similar way to model-based offline RL.
    '''

    EnvTrainTestSetup_params = luigi.DictParameter()
    compute_cv_score = luigi.Parameter(default=False)

    def run_task(self, input_list):

        working_dir = self._working_dir

        self.set_seed(self.EnvTrainTestSetup_params['seed'])
        _, train_episode_memory_dict = input_list[0]
        _, test_episode_memory_dict = input_list[1]
        episode_memory_dict = train_episode_memory_dict | test_episode_memory_dict
        env_train = input_list[2]
        qsar_model = env_train.qsar_model

        # learn QSAR model
        property_tuple_dict = {}
        key_list = list(episode_memory_dict.keys())

        seed = self.gen_seed()
        n_seed = max(len(key_list), self.EnvTrainTestSetup_params['n_splits'])
        if self.EnvTrainTestSetup_params['use_same_seed']:
            seed_list = [seed] * n_seed
        else:
            seed_list = self.seed_list(n_seed)

        # ground-truth property_tuple can be constructed w/o traiing
        try:
            key_list.remove('ground_truth')
            property_tuple_dict['ground_truth'] = PropertyTuple(
                qsar_model=input_list[2].qsar_model,
                episode_memory=episode_memory_dict['train']) # this episode memory is used only for training
        except ValueError:
            pass

        # test property_tuple requires much computation time, and should be executed first.
        try:
            key_list.remove('test')
            property_tuple_dict['test'] = _resample(0, 'test', episode_memory_dict['test'], seed,
                                                    working_dir, self.EnvTrainTestSetup_params)[1]
        except ValueError:
            pass

        if self.compute_cv_score:
            cv_score_list = self.cv_score(episode_memory_dict.get('train', None),
                                          seed_list,
                                          working_dir,
                                          n_splits=self.EnvTrainTestSetup_params['n_splits'])
            logger.info(' * cv{} score: {}'.format(self.EnvTrainTestSetup_params['n_splits'],
                                                   cv_score_list.mean()))
        else:
            cv_score_list = []

        property_tuple_dict.update(dict(Parallel(n_jobs=self.EnvTrainTestSetup_params['workers'],
                                                 verbose=10)([
                                                     delayed(_resample)(
                                                         each_idx % device_count(),
                                                         each_key,
                                                         episode_memory_dict[each_key],
                                                         seed_list[each_idx],
                                                         working_dir,
                                                         self.EnvTrainTestSetup_params)\
                                                     for each_idx, each_key in enumerate(key_list)])))

        # build an environment
        if 'train' in property_tuple_dict:
            env_train.qsar_model = property_tuple_dict['train'].qsar_model
        else:
            env_train.qsar_model = property_tuple_dict['ground_truth'].qsar_model
        return env_train, property_tuple_dict, cv_score_list

    def construct_dataset(self, episode_memory):
        mol_list = []
        tgt_list = []
        for each_transition in episode_memory.transition_list:
            if each_transition.done:
                mol_list.append(each_transition.state['current_mol'])
                tgt_list.append(each_transition.reward)
        return mol_list, tgt_list

    def cv_score(self, episode_memory, seed_list, working_dir, n_splits=5):
        ''' 
        '''
        if n_splits == 0:
            return torch.Tensor([])
        from sklearn.model_selection import KFold
        kf = KFold(n_splits)
        mol_list, tgt_list = episode_memory.extract_dataset()
        mol_list = np.array(mol_list)
        tgt_list = np.array(tgt_list)

        cv_score_list = Parallel(n_jobs=self.EnvTrainTestSetup_params['workers'],
                                 verbose=10)([
                                     delayed(_each_cv)(
                                         each_idx % device_count(),
                                         each_idx,
                                         mol_list[each_train_idx],
                                         tgt_list[each_train_idx],
                                         mol_list[each_test_idx],
                                         tgt_list[each_test_idx],
                                         seed_list[each_idx],
                                         working_dir,
                                         self.EnvTrainTestSetup_params)\
                                     for each_idx, (each_train_idx, each_test_idx) in enumerate(kf.split(mol_list,
                                                                                                         tgt_list))])
        return torch.stack(cv_score_list)
