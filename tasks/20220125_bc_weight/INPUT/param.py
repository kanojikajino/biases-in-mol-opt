#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Title '''

__author__ = 'Hiroshi Kajino <KAJINO@jp.ibm.com>'
__copyright__ = '(c) Copyright IBM Corp. 2020'

device = ['cpu', 'cuda'][1]
from numpy.random import default_rng

master_seed = 48
MAX_INT = 2 ** 31 - 1
use_same_seed = True
rng = default_rng(seed=master_seed)
radius = 2
activation = ['Softplus', 'ReLU'][0]
activation_kwargs = {'beta': 1.}

def seed_generator():
    return int(rng.integers(MAX_INT, size=1))


ReactantSelector_params = {
    'input_path': 'INPUT/data/enamine_building_blocks.csv',
    'n_mol': -1,
    'seed': seed_generator()
}

EnvSetup_params = {
    'qsar_model': [{'module': 'PenalizedLogP',
                    'kwargs': {}},
                   {'module': 'CCR5',
                    'kwargs': {}}][1],
    'data': {'reaction_tsv_path': 'INPUT/data/rxn_set.tsv'},
    'env': [{'module': 'GoalRewardReactionEnv',
             'kwargs': {'max_step': 1},
             'seed': seed_generator()},
            {'module': 'GoalRewardContinuousActionReactionEnv',
             'kwargs': {'n_lookup_reactants': 1,
                        'low': 0.0,
                        'high': 1.0,
                        'max_step': 1,
                        'device': device,
                        'fingerprint_kwargs': {'device': device}},
             'seed': seed_generator()}][1]}

OfflineTrainDataConstruction_params = {
    'gen_episode_memory_kwargs': {'n_episode': 1000,
                                  'remove_last': False,
                                  'include_offline_data': False,
                                  'workers': 16,
                                  'shutdown_everytime': False},
    'behavior_agent': [{'name': 'LegalRandomAgent',
                        'kwargs': {},
                        'seed': seed_generator()},
                       {'name': 'LegalRandomAgentWithOfflineData',
                        'seed': seed_generator(),
                        'kwargs': {'csv_path': 'INPUT/data/hiv_ccr5_chembl_274_ic50_clean.csv',
                                   'mol_col': 'SMILES',
                                   'tgt_col': 'pChEMBL Value',
                                   'preprocessing_list': ['lambda x: -math.log10(x)']}}][1],
    'seed': seed_generator()
}

OfflineTestDataConstruction_params = {
    'gen_test_episode_memory_kwargs': {'n_episode': 100000,
                                       'remove_last': False,
                                       'include_offline_data': False,
                                       'workers': 16,
                                       'shutdown_everytime': False},
    'behavior_agent': [{'name': 'LegalRandomAgent',
                        'kwargs': {},
                        'seed': seed_generator()},
                       {'name': 'LegalRandomAgentWithOfflineData',
                        'seed': seed_generator(),
                        'kwargs': {'csv_path': 'INPUT/data/hiv_ccr5_chembl_274_ic50_clean.csv',
                                   'mol_col': 'SMILES',
                                   'tgt_col': 'pChEMBL Value',
                                   'preprocessing_list': ['lambda x: -math.log10(x)']}}][1],
    'ground_truth': {'csv_path': 'INPUT/data/hiv_ccr5_chembl_274_ic50_clean.csv',
                     'mol_col': 'SMILES',
                     'tgt_col': 'pChEMBL Value',
                     'preprocessing_list': ['lambda x: -math.log10(x)']},
    'seed': seed_generator()
}


OfflineDataPreprocessing_params = {
    'eval_method': ['train_test_split', 'bootstrap', 'ground_truth'][1],
    'eval_kwargs': [{'test_ratio': 0.5}, {}][1],
    'seed': seed_generator(),
    'n_eval': 20
}


EnvTrainTestSetup_params = {
    'seed': seed_generator(),
    'use_same_seed': use_same_seed,
    'n_splits': 10,
    'fingerprint': [{'module': 'SelectedMolecularDescriptorsFingerprint',
                     'kwargs': {'device': device}},
                    {'module': 'DenseMorganFingerprint',
                     'kwargs': {'radius': radius,
                                'bit': 1024,
                                'use_chirality': True,
                                'device': device}}][1],
    'qsar_model': [{'module': 'MultiLayerPerceptronQSAR',
                    'kwargs': {'activation': activation,
                               'activation_kwargs': activation_kwargs,
                               'lmbd': 1e-8,
                               'out_dim_list': [96],
                               'activation_at_top': False,
                               'device': device}},
                   {'module': 'ExactGPQSAR',
                    'kwargs': {'covar_module_name': 'TanimotoJaccardKernel'}},
                   {'module': 'SklearnGPQSAR',
                    'kwargs': {}},
                   {'module': 'LinearDenseQSAR',
                    'kwargs': {'model_name': 'Ridge',
                               'model_kwargs': {'alpha': 1e-2}} #{'alphas': [2 ** x for x in range(-15, 0)], 'max_iter': 1e+5}}
                   }][0],
    'fit_kwargs': [{'max_update': 10000,
                    'batch_size': 128,
                    'optimizer': 'Adagrad',
                    'print_freq': 1000,
                    'optimizer_kwargs': {'lr': 1e-3}},
                   {'n_epochs': 200,
                    'optimizer_kwargs': {'lr': 4e-3}}][0],
    'workers': 12
}

FeatureSetup_params = {'workers': 1,
                       'fingerprint': [{'module': 'SparseMorganFingerprint',
                                        'kwargs': {'radius': radius,
                                                   'bit': 1024,
                                                   'use_chirality': True,
                                                   'use_zdd': True}},
                                       {'module': 'DenseMorganFingerprint',
                                        'kwargs': {'radius': radius,
                                                   'bit': 1024,
                                                   'use_chirality': True,
                                                   'device': device}},
                                       {'module': 'SelectedMolecularDescriptorsFingerprint',
                                        'kwargs': {'device': device}}][1],
                       'state_action_feature': [{'module': 'SingleSparseFingerprintFeature',
                                                 'kwargs': {}},
                                                {'module': 'ForwardSynthesisStateActionFeature',
                                                 'kwargs': {'device': device}}][1]
}

AgentSetup_params = {
    'use_same_seed': use_same_seed,
    'action_val_model': [{'module': 'LinearSparseActionValueFunction',
                          'kwargs': {}},
                         {'module': 'MLPActionValueFunction',
                          'kwargs': {'device': device,
                                     'seed': seed_generator(),
                                     'mlp_kwargs': {'out_dim_list': [256],
                                                    'activation': activation,
                                                    'activation_kwargs': activation_kwargs,
                                                    'out_dim': 1}}}][1],
    'agent': [{'module': ['LegalGreedyAgent', 'ExactFittedQIterationAgent'][1],
               'kwargs': {'discount': 1.0},
               'seed': seed_generator()},
              {'module': ['ForwardSynthesisActor', 'FiniteStepForwardSynthesisActor'][1],
               'kwargs': {'template_selector_kwargs': {'out_dim_list': [256],
                                                       'activation': activation,
                                                       'activation_kwargs': activation_kwargs,
                                                       'seed': seed_generator()},
                          'reactant_selector_kwargs': {'out_dim_list': [256],
                                                       'activation': activation,
                                                       'activation_kwargs': activation_kwargs,
                                                       'seed': seed_generator()},
                          'gumbel_softmax_kwargs': {'tau': 0.5, 'hard': False},
                          'device': device,
                          'discount': 1.0},
               'seed': seed_generator()}][1]
}

Train_params = {
    'fit_kwargs': {
        'n_epochs': 2000,
        'initial_episode_size': 128,
        'incremental_episode_size': 1,
        'tau': 0.005,
        'gumbel_softmax_annealing': 0.9999,
        'batch_size': 64,
        'episode_gen_freq': 10,
        'critic_update_freq': 1,
        'actor_update_freq': 2,
        'target_update_freq': 20,
        #'bc_weight': 1.,
        '@bc_weight': [2 ** x for x in range(4, -4, -1)],
        'burnin_period': 500,
        'print_freq': 100,
        'target_act_kwargs': {'std': 0.2, 'clip': 0.2},
        'optimizer_kwargs': {'lr': 4e-4}},
    'workers': 11,
    'seed': seed_generator(),
    'use_same_seed': use_same_seed
}

RunTest_params = {
    'n_trial': 1000,
    'workers': 10,
    'device': device,
    'seed': seed_generator()
}

TrainRewardModel4Eval_params = {
    'mode': ['skip', 'density_ratio', 'covariate_shift'][2],
    'weight_decay': 1.0,
    'density_ratio': {
        'fingerprint': [{'module': 'SelectedMolecularDescriptorsFingerprint',
                         'kwargs': {'device': device}},
                        {'module': 'DenseMorganFingerprint',
                         'kwargs': {'radius': radius,
                                    'bit': 1024,
                                    'use_chirality': True}},
                        {'module': 'DenseNeuralFingerprint',
                         'kwargs': {'fingerprint_kwargs': {'radius': 2,
                                                           'bit': 1024,
                                                           'use_chirality': True,
                                                           'device': device},
                                    'qsar_model_kwargs': {'activation': activation,
                                                          'activation_kwargs': activation_kwargs,
                                                          'activation_at_top': False,
                                                          'lmbd': 1e-6,
                                                          'out_dim_list': [96],
                                                          'device': device},
                                    'device': device,
                                    'seed': seed_generator()},
                         'fit_kwargs': {'max_update': 10000,
                                        'batch_size': 256,
                                        'optimizer': 'Adagrad',
                                        'print_freq': 1000,
                                        'optimizer_kwargs': {'lr': 1e-3}}}][2],
        'covshift': {'module': ['KernelMeanMatching', 'KernelULSIF'][1],
                     'kwargs': [{'kernel_name': ['RBFKernel', 'TanimotoSimilarity', 'LinearKernel'][2],
                                 'kernel_kwargs': {},
                                 'max_beta': 1e+2, #'lmbd': 1e+3
                                 'tol': 1e-5},
                                {'kernel_name': ['RBFKernel', 'TanimotoSimilarity', 'LinearKernel'][2],
                                 'kernel_kwargs': {},
                                 'lmbd_list': [2 ** x for x in range(-20, 5)]}][1]}},
    'workers': 16,
    'seed': seed_generator()
}

# TrainDensityRatio_params = {
#     'skip': False,
#     'fingerprint': [{'module': 'SelectedMolecularDescriptorsFingerprint',
#                      'kwargs': {'device': device}},
#                     {'module': 'DenseMorganFingerprint',
#                      'kwargs': {'radius': radius,
#                                 'bit': 1024,
#                                 'use_chirality': True}}][1],
#     'covshift': {'module': ['KernelMeanMatching', 'KernelULSIF'][1],
#                  'kwargs': [{'kernel_name': ['RBFKernel', 'TanimotoSimilarity'][1],
#                             'kernel_kwargs': {},
#                             'max_beta': 1e+2, #'lmbd': 1e+3
#                              'tol': 1e-5},
#                             {'kernel_name': ['RBFKernel', 'TanimotoSimilarity'][1],
#                              'kernel_kwargs': {},
#                              'lmbd_list': [2 ** x for x in range(-20, 0)]}][1]},
#     'workers': 16
# }


PerformanceEvaluation_params = {'summary_kwargs': {'apply_decision_rule': False,
                                                   'threshold': 1.0},
                                'workers': 11,
                                'doubly_robust': {'@importance_sampling': [False, True],
                                                  'direct_method': True},
                                '@covariate_shift': [False, True]
}


MultipleRun_params = {
    'ReactantSelector_params': ReactantSelector_params,
    'EnvSetup_params': EnvSetup_params,
    'OfflineTrainDataConstruction_params': OfflineTrainDataConstruction_params,
    'OfflineTestDataConstruction_params': OfflineTestDataConstruction_params,
    'OfflineTrainDataPreprocessing_params': OfflineDataPreprocessing_params,
    'OfflineTestDataPreprocessing_params': OfflineDataPreprocessing_params,
    'EnvTrainTestSetup_params': EnvTrainTestSetup_params,
    'FeatureSetup_params': FeatureSetup_params,
    'AgentSetup_params': AgentSetup_params,
    'Train_params': Train_params,
    'TrainRewardModel4Eval_params': TrainRewardModel4Eval_params,
    'RunTest_params': RunTest_params,
    'PerformanceEvaluation_params': PerformanceEvaluation_params,
    'seed': master_seed,
    'n_iter': 5
}

PlotBiasEstimation_params = {
    'x': ('Train_params', 'fit_kwargs', 'bc_weight'),
    'plot_config_list': [{'col_name': 'mean_train_return_---',
                          'yerr_col_name': 'std_train_return_---',
                          'plot_kwargs': {'marker': 'o', 'logx': True, 'color': 'tab:blue'}},
                         {'col_name': 'mean_train_return_--+',
                          'yerr_col_name': 'std_train_return_--+',
                          'plot_kwargs': {'marker': '*', 'logx': True, 'color': 'tab:orange'}},
                         {'col_name': 'mean_train_return_+--',
                          'yerr_col_name': 'std_train_return_+--',
                          'plot_kwargs': {'marker': '+', 'logx': True, 'color': 'tab:green'}},
                         #{'col_name': 'mean_train_return_-+-',
                         # 'yerr_col_name': 'std_train_return_-+-',
                         # 'plot_kwargs': {'marker': 'x', 'logx': True, 'color': 'tab:brown'}},
                         #{'col_name': 'mean_train_return_+-+',
                         # 'yerr_col_name': 'std_train_return_+-+',
                         # 'plot_kwargs': {'marker': '<', 'logx': True}},
                         {'col_name': 'mean_test_return_with_mis_bias',
                          'yerr_col_name': 'std_test_return_with_mis_bias',
                          'plot_kwargs': {'marker': 'v', 'logx': True, 'color': 'tab:red'}},
                         {'col_name': 'mean_ground_truth_return',
                          'yerr_col_name': 'std_ground_truth_return',
                          'plot_kwargs': {'marker': '^', 'logx': True, 'color': 'tab:purple'}}
    ],
    'fig_config': {'xlabel': {'xlabel': r'Behavior cloning coefficient $\nu$',
                              'fontsize': 16},
                   'ylabel': {'ylabel': 'Performance',
                              'fontsize': 16},
                   'legend': {'labels': [r'$J_{\mathrm{PI}}(\hat{G},\hat{f})$',
                                         r'$J_{\mathrm{PI}}$ w/ bootstrap bias reduction', #r'$J_{\mathrm{PI}}^{-+}$',
                                         r'$J_{\mathrm{PI}}$ w/ covariate shift adaptation', #r'$J_{\mathrm{PI}}^{+-}$',
                                         #r'$J_{\mathrm{DR}}$',
                                         #r'$J_{\mathrm{PI}}^{++}$',
                                         r'$J_{\mathrm{PI}}(\hat{G}, f^{\infty})$',
                                         r'$J^\star(\hat{G})$'],
                              'fontsize': 12,
                              'loc': 'upper center',
                              'bbox_to_anchor': (0.45, -0.13),
                              'ncol': 2}}
}
