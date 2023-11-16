#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "Copyright IBM Corp. 2021"

from copy import deepcopy
import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from mlbasics.nn import MultiLayerPerceptron
from mlbasics.utils import TorchRngMixin
from .base import AgentBase
from ..feature import DenseMorganFingerprintStateFeature


class ForwardSynthesisActor(TorchRngMixin, AgentBase):

    def __init__(self,
                 action_val_model,
                 seed,
                 action_space,
                 max_step,
                 template_selector_kwargs={},
                 reactant_selector_kwargs={},
                 gumbel_softmax_kwargs={},
                 discount=1.0,
                 device='cpu',
                 **kwargs):
        super().__init__(seed,
                         discount,
                         max_step,
                         device=device,
                         **kwargs)
        self.set_torch_seed(seed)
        from molgym.chem.fingerprint import SparseMorganFingerprint
        self.obs2feature = DenseMorganFingerprintStateFeature(
            fingerprint=SparseMorganFingerprint(radius=2,
                                                bit=1024,
                                                device=device),
            device=device)

        self.critic_1 = action_val_model
        self.critic_2 = deepcopy(action_val_model)
        self.critic_2.reset_parameters()
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.template_selector = MultiLayerPerceptron(
            in_dim=self.obs2feature.out_dim,
            out_dim=action_space[0].n,
            device=self.device,
            **template_selector_kwargs)
        self.target_template_selector = deepcopy(self.template_selector)

        self.reactant_selector = MultiLayerPerceptron(
            in_dim=self.obs2feature.out_dim+action_space[0].n,
            out_dim=action_space[1].shape[0],
            device=self.device,
            **reactant_selector_kwargs)
        self.target_reactant_selector = deepcopy(self.reactant_selector)

        self.gumbel_softmax_kwargs = dict(gumbel_softmax_kwargs)
        self.template_selector_loss = nn.CrossEntropyLoss()

    def to(self, device):
        super().to(device)
        self.obs2feature.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)
        self.template_selector.to(device)
        self.target_template_selector.to(device)
        self.reactant_selector.to(device)
        self.target_reactant_selector.to(device)
        self.delete_rng()

    def delete_rng(self):
        super().delete_rng()
        self.critic_1.delete_rng()
        self.critic_2.delete_rng()
        self.target_critic_1.delete_rng()
        self.target_critic_2.delete_rng()
        self.template_selector.delete_rng()
        self.target_template_selector.delete_rng()
        self.reactant_selector.delete_rng()
        self.target_reactant_selector.delete_rng()
        

    def init_optimizer(self, optimizer='Adagrad', optimizer_kwargs={'lr': 1e-2}):
        self.template_selector_optimizer = getattr(torch.optim, optimizer)(
            params=self.template_selector.parameters(),
            **optimizer_kwargs)
        self.reactant_selector_optimizer = getattr(torch.optim, optimizer)(
            params=self.reactant_selector.parameters(),
            **optimizer_kwargs)

    def update_target(self, tau):
        def _update_target(current_module, tgt_module, tau):
            for each_param, each_tgt_param in zip(current_module.parameters(),
                                                  tgt_module.parameters()):
                each_tgt_param.data.copy_(tau * each_param.data
                                          + (1 - tau) * each_tgt_param.data)
        _update_target(self.critic_1, self.target_critic_1, tau)
        _update_target(self.critic_2, self.target_critic_2, tau)
        _update_target(self.template_selector,
                       self.target_template_selector,
                       tau)
        _update_target(self.reactant_selector,
                       self.target_reactant_selector,
                       tau)

    def _act(self, observation, action_space):
        return self.base_act(observation, action_space)

    def criticize(self, observation, action, action_space):
        return self.critic_1.forward(observation,
                                     action,
                                     action_space)

    def soft_act(self, observation, action_space):
        return self.base_act(observation, action_space, hard=False)

    def target_act(self, observation, action_space, std, clip):
        return self.base_act(observation, action_space,
                             std=std, clip=clip, target=True)

    def base_act(self,
                 observation,
                 action_space,
                 device=None,
                 std=0,
                 clip=float('inf'),
                 target=False,
                 hard=True):
        if device is None:
            device = self.device
        if target:
            template_selector = self.target_template_selector
            reactant_selector = self.target_reactant_selector
        else:
            template_selector = self.template_selector
            reactant_selector = self.reactant_selector

        legal_reaction_template = action_space.reaction_corpus.legal_reaction_template(
            observation['current_mol'])
        if not legal_reaction_template:
            return action_space.stop_action
        legal_reaction_idx_list = sorted(list(set(each_reaction[0]
                                                  for each_reaction
                                                  in legal_reaction_template)))
        mask = -float('inf') * torch.ones(action_space[0].n, device=self.device)
        mask[torch.tensor(legal_reaction_idx_list)] = 0.0
        feature_array = self.obs2feature.forward(observation=observation)
        template_logit = template_selector.forward(feature_array)
        template_logit = template_logit + mask
        template_hard = gumbel_softmax(template_logit, hard=True)
        template_soft = gumbel_softmax(template_logit,
                                       **self.gumbel_softmax_kwargs)
        action_array = reactant_selector(torch.cat([feature_array, template_soft]))
        if std != 0:
            action_array = action_array \
                + (std * torch.randn(size=action_array.shape,
                                     dtype=action_array.dtype,
                                     device=self.device,
                                     generator=self.torch_rng)).clip(-clip, clip)
        if hard:
            return template_hard.to(device), action_array.to(device)
        return template_soft.to(device), action_array.to(device)

    def fit(self,
            episode_memory,
            env,
            n_epochs=1000,
            initial_episode_size=100,
            incremental_episode_size=100,
            batch_size=10,
            episode_gen_freq=100,
            critic_update_freq=1,
            actor_update_freq=10,
            target_update_freq=10,
            tau=0.01,
            gumbel_softmax_annealing=0.95,
            target_act_kwargs={'std': 0.2, 'clip': 0.5},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 3e-4},
            bc_weight=1e-1,
            burnin_period=0,
            print_freq=1,
            logger=print,
            **kwargs):
        ''' train the actor using `env`

        Parameters
        ----------
        bc_weight : float
            weight for behavior cloning
        '''
        action_space = env.action_space
        self.critic_1.init_optimizer(optimizer, optimizer_kwargs)
        self.critic_2.init_optimizer(optimizer, optimizer_kwargs)
        self.init_optimizer(optimizer, optimizer_kwargs)
        main_episode_memory = self.gen_episode_memory(env,
                                                      initial_episode_size,
                                                      random=True)
        torch.manual_seed(self.gen_seed())
        critic_1_loss = None
        critic_2_loss = None
        actor_loss = None
        template_selector_loss = None

        for each_epoch in range(n_epochs):
            if each_epoch % episode_gen_freq == 0:
                gen_episode_memory = self.gen_episode_memory(env, incremental_episode_size, shutdown_everytime=True)
                #main_episode_memory = main_episode_memory + episode_memory
            #minibatch_transition_list = main_episode_memory.sample_minibatch(batch_size)
            gen_minibatch_transition_list = gen_episode_memory.sample_minibatch(batch_size)
            minibatch_transition_list = episode_memory.sample_minibatch(batch_size)
            if each_epoch % critic_update_freq == 0:
                state_action_train_list, y_train \
                    = self.construct_critic_dataset(gen_minibatch_transition_list,
                                                    action_space,
                                                    target_act_kwargs)
                critic_1_loss = self.critic_1.fit(state_action_train_list,
                                                  y_train,
                                                  action_space,
                                                  n_epochs=1)
                critic_2_loss = self.critic_2.fit(state_action_train_list,
                                                  y_train,
                                                  action_space,
                                                  n_epochs=1)
            if each_epoch % actor_update_freq == 0 and each_epoch > burnin_period:
                actor_loss = self.actor_fit(gen_minibatch_transition_list,
                                            minibatch_transition_list,
                                            action_space,
                                            bc_weight,
                                            n_epochs=1)
                template_selector_loss = self.template_selector_fit(gen_minibatch_transition_list,
                                                                    action_space,
                                                                    n_epochs=1)
            if each_epoch % target_update_freq == 0:
                self.update_target(tau)
            if each_epoch % print_freq == 0:
                logger(' iter: {}'.format(each_epoch))
                logger(' * critic_loss : {}, {}'.format(critic_1_loss, critic_2_loss))
                logger(' * actor_loss : {}'.format(actor_loss))
                logger(' * template_selector_loss : {}'.format(template_selector_loss))
                logger(' * gs temperature : {}'.format(self.gumbel_softmax_kwargs['tau']))
            self.gumbel_softmax_kwargs['tau'] *= gumbel_softmax_annealing

    def actor_fit(self,
                  gen_transition_list,
                  behavior_transition_list,
                  action_space,
                  bc_weight,
                  n_epochs=10):
        for _ in range(n_epochs):
            gen_state_action_list = [(each_transition.state,
                                      self.soft_act(each_transition.state, action_space))
                                     for each_transition in gen_transition_list]
            loss = -self.critic_1.batch_forward(gen_state_action_list, action_space).sum()
            behavior_pred_true_action_list = [(each_transition.action,
                                               self.soft_act(each_transition.state, action_space))
                                              for each_transition in behavior_transition_list\
                                              if each_transition.action[0] is not None]
            template_true_array = torch.stack([each_pred_true_tuple[0][0]
                                               for each_pred_true_tuple in behavior_pred_true_action_list]).to(self.device)
            template_pred_array = torch.stack([each_pred_true_tuple[1][0]
                                               for each_pred_true_tuple in behavior_pred_true_action_list])
            reactant_true_array = torch.stack([each_pred_true_tuple[0][1]
                                               for each_pred_true_tuple in behavior_pred_true_action_list]).to(self.device)
            reactant_pred_array = torch.stack([each_pred_true_tuple[1][1]
                                               for each_pred_true_tuple in behavior_pred_true_action_list])
            mask_array = template_pred_array != 0
            loss = loss + 0.5 * bc_weight * torch.mean((reactant_true_array - reactant_pred_array) ** 2)\
                - 0.5 * bc_weight * torch.mean(torch.log(template_pred_array[mask_array]) * template_true_array[mask_array])
            self.template_selector_optimizer.zero_grad()
            self.reactant_selector_optimizer.zero_grad()
            loss.backward()
            self.template_selector_optimizer.step()
            self.reactant_selector_optimizer.step()
        return loss.item() / len(gen_transition_list)

    def template_selector_fit(self,
                              transition_list,
                              action_space,
                              n_epochs=10):
        feature_list = [self.obs2feature(observation=each_transition.state)
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
        feature_tensor = torch.stack(feature_list)
        for _ in range(n_epochs):
            tgt_list = [torch.where(each_transition.action[0] == 1.0)[0]
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
            pred_logit = self.template_selector.batch_forward(feature_tensor)
            tgt_tensor = torch.cat(tgt_list).to(self.device)
            loss = self.template_selector_loss(pred_logit, tgt_tensor)
            self.template_selector_optimizer.zero_grad()
            loss.backward()
            self.template_selector_optimizer.step()
        return loss.item() / len(transition_list)

    def construct_critic_dataset(self,
                                 transition_list,
                                 action_space,
                                 target_act_kwargs):
        state_action_train_list = []
        reward_list = []
        next_state_action_train_list = []
        y_train_list = []
        for each_transition in transition_list:
            state_action_train_list.append((each_transition.state,
                                            each_transition.action))
            with torch.no_grad():
                next_action = self.target_act(each_transition.next_state,
                                              action_space,
                                              **target_act_kwargs)
            next_state_action_train_list.append((each_transition.next_state,
                                                 next_action))
            reward_list.append(each_transition.reward)
        with torch.no_grad():
            target_q = torch.min(
                torch.stack([self.target_critic_1.batch_forward(next_state_action_train_list,
                                                                action_space),
                             self.target_critic_2.batch_forward(next_state_action_train_list,
                                                                action_space)]),
                dim=0)[0]
            y_train = torch.tensor(reward_list, device=self.device) + self.discount * target_q
        return state_action_train_list, y_train


class CriticLast(nn.Module):

    def __init__(self, env, device):
        super().__init__()
        self.reward_model = env.reward_model
        self.device = device

    def forward(self, observation, action, action_space):
        return self.reward_model(observation,
                                 action,
                                 True,
                                 observation['current_mol'])

    def batch_forward(self, obs_action_list, action_space):
        forward_list = [self.forward(each_obs, each_action, action_space)
                        for each_obs, each_action
                        in obs_action_list]
        return torch.tensor(forward_list, device=self.device)

    def delete_rng(self):
        pass

    def to(self, device):
        self.reward_model.to(device)
        self.device = device


class NullModule(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, observation, action, action_space):
        return 0.0
        
    def batch_forward(self, obs_action_list, action_space):
        forward_list = [self.forward(each_obs, each_action, action_space)
                        for each_obs, each_action
                        in obs_action_list]
        return torch.tensor(forward_list, device=self.device)

    def delete_rng(self):
        pass

    def to(self, device):
        self.device = device

class FiniteStepForwardSynthesisActor(TorchRngMixin, AgentBase):

    def __init__(self,
                 action_val_model,
                 seed,
                 action_space,
                 max_step,
                 template_selector_kwargs={},
                 reactant_selector_kwargs={},
                 gumbel_softmax_kwargs={},
                 discount=1.0,
                 device='cpu',
                 **kwargs):
        super().__init__(seed,
                         discount,
                         max_step,
                         device=device,
                         **kwargs)
        self.set_torch_seed(seed)
        from molgym.chem.fingerprint import SparseMorganFingerprint
        self.obs2feature = DenseMorganFingerprintStateFeature(
            fingerprint=SparseMorganFingerprint(radius=2,
                                                bit=1024,
                                                device=device),
            device=device)

        self.critic_1_list = nn.ModuleList([deepcopy(action_val_model) for _ in range(max_step+1)])
        self.critic_2_list = nn.ModuleList([deepcopy(action_val_model) for _ in range(max_step+1)])
        for each_step in range(max_step+1):
            self.critic_1_list[each_step].reset_parameters()
            self.critic_2_list[each_step].reset_parameters()

        self.template_selector_list = nn.ModuleList([
            MultiLayerPerceptron(
                in_dim=self.obs2feature.out_dim,
                out_dim=action_space[0].n,
                device=self.device,
                **template_selector_kwargs) for _ in range(max_step+1)])

        self.reactant_selector_list = nn.ModuleList([
            MultiLayerPerceptron(
                in_dim=self.obs2feature.out_dim+action_space[0].n,
                out_dim=action_space[1].shape[0],
                device=self.device,
                **reactant_selector_kwargs) for _ in range(max_step+1)])

        self.gumbel_softmax_kwargs = dict(gumbel_softmax_kwargs)
        self.template_selector_loss = nn.CrossEntropyLoss()


    def to(self, device):
        super().to(device)
        self.obs2feature.to(device)
        self.critic_1_list.to(device)
        self.critic_2_list.to(device)
        self.template_selector_list.to(device)
        self.reactant_selector_list.to(device)
        self.delete_rng()

    def delete_rng(self):
        super().delete_rng()
        for each_step in range(self.max_step+1):
            self.critic_1_list[each_step].delete_rng()
            self.critic_2_list[each_step].delete_rng()
            self.template_selector_list[each_step].delete_rng()
            self.reactant_selector_list[each_step].delete_rng()
        

    def init_optimizer(self, optimizer='Adagrad', optimizer_kwargs={'lr': 1e-2}):
        self.template_selector_optimizer_list \
            = [getattr(torch.optim, optimizer)(
                params=self.template_selector_list[each_step].parameters(),
                **optimizer_kwargs)
               for each_step in range(self.max_step+1)]
        self.reactant_selector_optimizer_list \
            = [getattr(torch.optim, optimizer)(
                params=self.reactant_selector_list[each_step].parameters(),
                **optimizer_kwargs)
               for each_step in range(self.max_step+1)]

    def _act(self, observation, action_space):
        return self.base_act(observation, action_space)

    def criticize(self, observation, action, action_space):
        return self.critic_1_list[observation['n_step']].forward(
            observation,
            action,
            action_space)

    def soft_act(self, observation, action_space):
        return self.base_act(observation, action_space, hard=False)

    def target_act(self, observation, action_space, std, clip):
        return self.base_act(observation, action_space,
                             std=std, clip=clip, target=True)

    def base_act(self,
                 observation,
                 action_space,
                 device=None,
                 std=0,
                 clip=float('inf'),
                 hard=True,
                 target=False):
        if observation['n_step'] == self.max_step:
            return action_space.stop_action
        if device is None:
            device = self.device
        #if target:
        #    template_selector = self.template_selector_list[observation['n_step']+1]
        #    reactant_selector = self.reactant_selector_list[observation['n_step']+1]
        #else:
        template_selector = self.template_selector_list[observation['n_step']]
        reactant_selector = self.reactant_selector_list[observation['n_step']]

        legal_reaction_template = action_space.reaction_corpus.legal_reaction_template(
            observation['current_mol'])
        if not legal_reaction_template:
            return action_space.stop_action
        legal_reaction_idx_list = sorted(list(set(each_reaction[0]
                                                  for each_reaction
                                                  in legal_reaction_template)))
        mask = -float('inf') * torch.ones(action_space[0].n, device=self.device)
        mask[torch.tensor(legal_reaction_idx_list)] = 0.0
        feature_array = self.obs2feature.forward(observation=observation)
        template_logit = template_selector.forward(feature_array)
        template_logit = template_logit + mask
        template_hard = gumbel_softmax(template_logit, hard=True)
        template_soft = gumbel_softmax(template_logit,
                                       **self.gumbel_softmax_kwargs)
        action_array = reactant_selector(torch.cat([feature_array, template_soft]))
        if std != 0:
            action_array = action_array \
                + (std * torch.randn(size=action_array.shape,
                                     dtype=action_array.dtype,
                                     device=self.device,
                                     generator=self.torch_rng)).clip(-clip, clip)
        if hard:
            return template_hard.to(device), action_array.to(device)
        return template_soft.to(device), action_array.to(device)

    def fit(self,
            episode_memory,
            env,
            n_epochs=1000,
            n_inner_epochs=10,
            batch_size=10,
            actor_update_freq=10,
            episode_gen_freq=1,
            incremental_episode_size=1,
            gumbel_softmax_annealing=0.95,
            target_act_kwargs={'std': 0.2, 'clip': 0.5},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 3e-4},
            bc_weight=0.,
            burnin_period=0,
            print_freq=1,
            logger=print,
            **kwargs):
        ''' train the actor using `env`

        Parameters
        ----------
        bc_weight : float
            weight for behavior cloning
        '''
        tau_init = self.gumbel_softmax_kwargs['tau']
        action_space = env.action_space
        for each_step in range(self.max_step+1):
            self.critic_1_list[each_step].init_optimizer(optimizer, optimizer_kwargs)
            self.critic_2_list[each_step].init_optimizer(optimizer, optimizer_kwargs)
        self.init_optimizer(optimizer, optimizer_kwargs)

        torch.manual_seed(self.gen_seed())
        critic_1_loss = None
        critic_2_loss = None
        actor_loss = None
        template_selector_loss = None

        self.critic_1_list[self.max_step] = CriticLast(env, self.device)
        self.critic_2_list[self.max_step] = CriticLast(env, self.device)

        for each_epoch in range(n_epochs):
            for each_step in range(self.max_step-1, -1, -1):
                self.gumbel_softmax_kwargs['tau'] = tau_init
                if each_epoch % episode_gen_freq == 0 and episode_gen_freq != -1:
                    gen_episode_memory = self.gen_episode_memory(env, incremental_episode_size, shutdown_everytime=True)
                    episode_memory.append(gen_episode_memory)
                for each_inner_epoch in range(n_inner_epochs * each_epoch, n_inner_epochs * (each_epoch+1)):
                    self.gumbel_softmax_kwargs['tau'] = tau_init * (gumbel_softmax_annealing ** each_inner_epoch)
                    minibatch_transition_list = episode_memory.sample_minibatch(batch_size, each_step)
                    state_action_train_list, y_train \
                        = self.construct_critic_dataset(minibatch_transition_list,
                                                        action_space,
                                                        target_act_kwargs)
                    critic_1_loss = self.critic_1_list[each_step].fit(
                        state_action_train_list,
                        y_train,
                        action_space,
                        n_epochs=1)
                    critic_2_loss = self.critic_2_list[each_step].fit(
                        state_action_train_list,
                        y_train,
                        action_space,
                        n_epochs=1)
                    if each_inner_epoch % actor_update_freq == 0 and each_inner_epoch > burnin_period:
                        if each_step != self.max_step:
                            actor_loss = self.actor_fit(minibatch_transition_list,
                                                        each_step,
                                                        action_space,
                                                        bc_weight,
                                                        n_epochs=1)
                            template_selector_loss = self.template_selector_fit(minibatch_transition_list,
                                                                                each_step,
                                                                                action_space,
                                                                                n_epochs=1)
                    if each_inner_epoch % print_freq == 0:
                        logger(' iter: {}'.format(each_inner_epoch))
                        logger(' * critic_loss : {}, {}'.format(critic_1_loss, critic_2_loss))
                        logger(' * actor_loss : {}'.format(actor_loss))
                        logger(' * template_selector_loss : {}'.format(template_selector_loss))
                        logger(' * gs temperature : {}'.format(self.gumbel_softmax_kwargs['tau']))
        self.critic_1_list[self.max_step] = NullModule(self.device)
        self.critic_2_list[self.max_step] = NullModule(self.device)


    def actor_fit(self,
                  transition_list,
                  n_step,
                  action_space,
                  bc_weight,
                  n_epochs=10):
        for _ in range(n_epochs):
            state_action_list = [(each_transition.state,
                                  self.soft_act(each_transition.state, action_space))
                                 for each_transition in transition_list]
            loss = -self.critic_1_list[n_step].batch_forward(state_action_list, action_space).mean()
            pred_true_action_list = [(each_transition.action,
                                      self.soft_act(each_transition.state, action_space))
                                     for each_transition in transition_list\
                                     if each_transition.action[0] is not None]
            template_true_array = torch.stack([each_pred_true_tuple[0][0]
                                               for each_pred_true_tuple in pred_true_action_list]).to(self.device)
            template_pred_array = torch.stack([each_pred_true_tuple[1][0]
                                               for each_pred_true_tuple in pred_true_action_list])
            reactant_true_array = torch.stack([each_pred_true_tuple[0][1]
                                               for each_pred_true_tuple in pred_true_action_list]).to(self.device)
            reactant_pred_array = torch.stack([each_pred_true_tuple[1][1]
                                               for each_pred_true_tuple in pred_true_action_list])
            loss = loss + 0.5 * bc_weight * torch.mean(torch.log(reactant_pred_array + 1e-8) * reactant_true_array)\
                - 0.5 * bc_weight * torch.mean(torch.log(template_pred_array + 1e-8) * template_true_array)
            self.template_selector_optimizer_list[n_step].zero_grad()
            self.reactant_selector_optimizer_list[n_step].zero_grad()
            loss.backward()
            self.template_selector_optimizer_list[n_step].step()
            self.reactant_selector_optimizer_list[n_step].step()
        return loss.item()

    def template_selector_fit(self,
                              transition_list,
                              n_step,
                              action_space,
                              n_epochs=10):
        feature_list = [self.obs2feature(observation=each_transition.state)
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
        feature_tensor = torch.stack(feature_list)
        for _ in range(n_epochs):
            tgt_list = [torch.where(each_transition.action[0] == 1.0)[0]
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
            pred_logit = self.template_selector_list[n_step].batch_forward(feature_tensor)
            tgt_tensor = torch.cat(tgt_list).to(self.device)
            loss = self.template_selector_loss(pred_logit, tgt_tensor)
            self.template_selector_optimizer_list[n_step].zero_grad()
            loss.backward()
            self.template_selector_optimizer_list[n_step].step()
        return loss.item()

    def construct_critic_dataset(self,
                                 transition_list,
                                 action_space,
                                 target_act_kwargs):
        state_action_train_list = []
        reward_list = []
        next_state_action_train_list = []
        y_train_list = []
        for each_transition in transition_list:
            state_action_train_list.append((each_transition.state,
                                            each_transition.action))
            if each_transition.state['n_step'] == self.max_step:
                next_action = action_space.stop_action
            else:
                with torch.no_grad():
                    next_action = self.target_act(each_transition.next_state,
                                                  action_space,
                                                  **target_act_kwargs)
            next_state_action_train_list.append((each_transition.next_state,
                                                 next_action))
            reward_list.append(each_transition.reward)
        with torch.no_grad():
            if each_transition.state['n_step'] == self.max_step:
                target_q = 0.
            else:
                target_q = torch.min(
                    torch.stack([self.critic_1_list[each_transition.state['n_step']+1].batch_forward(
                        next_state_action_train_list,
                        action_space),
                                 self.critic_2_list[each_transition.state['n_step']+1].batch_forward(
                                     next_state_action_train_list,
                                     action_space)]),
                    dim=0)[0]
            y_train = torch.tensor(reward_list, device=self.device) + self.discount * target_q
        return state_action_train_list, y_train



class FiniteStepActorCritic(TorchRngMixin, AgentBase):

    def __init__(self,
                 action_val_model,
                 seed,
                 action_space,
                 max_step,
                 template_selector_kwargs={},
                 reactant_selector_kwargs={},
                 gumbel_softmax_kwargs={},
                 discount=1.0,
                 device='cpu',
                 **kwargs):
        super().__init__(seed,
                         discount,
                         max_step,
                         device=device,
                         **kwargs)
        self.set_torch_seed(seed)
        from molgym.chem.fingerprint import SparseMorganFingerprint
        self.obs2feature = DenseMorganFingerprintStateFeature(
            fingerprint=SparseMorganFingerprint(radius=2,
                                                bit=1024,
                                                device=device),
            device=device)

        self.critic_1_list = nn.ModuleList([deepcopy(action_val_model) for _ in range(max_step+1)])
        self.critic_2_list = nn.ModuleList([deepcopy(action_val_model) for _ in range(max_step+1)])
        for each_step in range(max_step+1):
            self.critic_1_list[each_step].reset_parameters()
            self.critic_2_list[each_step].reset_parameters()

        self.template_selector_list = nn.ModuleList([
            MultiLayerPerceptron(
                in_dim=self.obs2feature.out_dim,
                out_dim=action_space[0].n,
                device=self.device,
                **template_selector_kwargs) for _ in range(max_step+1)])

        self.reactant_selector_list = nn.ModuleList([
            MultiLayerPerceptron(
                in_dim=self.obs2feature.out_dim+action_space[0].n,
                out_dim=action_space[1].n,
                device=self.device,
                **reactant_selector_kwargs) for _ in range(max_step+1)])

        self.gumbel_softmax_kwargs = dict(gumbel_softmax_kwargs)
        self.template_selector_loss = nn.CrossEntropyLoss()
        self.reactant_selector_loss = nn.CrossEntropyLoss()

    def to(self, device):
        super().to(device)
        self.obs2feature.to(device)
        self.critic_1_list.to(device)
        self.critic_2_list.to(device)
        self.template_selector_list.to(device)
        self.reactant_selector_list.to(device)
        self.delete_rng()

    def delete_rng(self):
        super().delete_rng()
        for each_step in range(self.max_step+1):
            self.critic_1_list[each_step].delete_rng()
            self.critic_2_list[each_step].delete_rng()
            self.template_selector_list[each_step].delete_rng()
            self.reactant_selector_list[each_step].delete_rng()

    def init_optimizer(self, optimizer='Adagrad', optimizer_kwargs={'lr': 1e-2}):
        self.template_selector_optimizer_list \
            = [getattr(torch.optim, optimizer)(
                params=self.template_selector_list[each_step].parameters(),
                **optimizer_kwargs)
               for each_step in range(self.max_step+1)]
        self.reactant_selector_optimizer_list \
            = [getattr(torch.optim, optimizer)(
                params=self.reactant_selector_list[each_step].parameters(),
                **optimizer_kwargs)
               for each_step in range(self.max_step+1)]

    def _act(self, observation, action_space):
        return self.base_act(observation, action_space)

    def criticize(self, observation, action, action_space):
        return self.critic_1_list[observation['n_step']].forward(
            observation,
            action,
            action_space)

    def soft_act(self, observation, action_space):
        return self.base_act(observation, action_space, hard=False)

    def target_act(self, observation, action_space, std, clip):
        return self.base_act(observation, action_space,
                             std=std, clip=clip, target=True)

    def base_act(self,
                 observation,
                 action_space,
                 device=None,
                 std=0,
                 clip=float('inf'),
                 hard=True,
                 target=False):
        if observation['n_step'] == self.max_step:
            return action_space.stop_action
        if device is None:
            device = self.device
        #if target:
        #    template_selector = self.template_selector_list[observation['n_step']+1]
        #    reactant_selector = self.reactant_selector_list[observation['n_step']+1]
        #else:
        template_selector = self.template_selector_list[observation['n_step']]
        reactant_selector = self.reactant_selector_list[observation['n_step']]

        legal_reaction_template = action_space.reaction_corpus.legal_reaction_template(
            observation['current_mol'])
        if not legal_reaction_template:
            return action_space.stop_action
        legal_reaction_idx_list = sorted(list(set(each_reaction[0]
                                                  for each_reaction
                                                  in legal_reaction_template)))
        mask = -float('inf') * torch.ones(action_space[0].n, device=self.device)
        mask[torch.tensor(legal_reaction_idx_list)] = 0.0
        feature_array = self.obs2feature.forward(observation=observation)

        template_logit = template_selector.forward(feature_array)
        template_logit = template_logit + mask
        template_hard = gumbel_softmax(template_logit, hard=True)
        template_soft = gumbel_softmax(template_logit,
                                       **self.gumbel_softmax_kwargs)
        template_idx = torch.where(template_hard == 1)[0].item()

        reactant_logit = reactant_selector.forward(torch.cat([feature_array, template_soft]))
        mask = -float('inf') * torch.ones(action_space[1].n, device=self.device)
        reactant_idx_list = []
        for each_list in action_space.reaction_corpus.legal_reactant_list[template_idx]:
            reactant_idx_list.extend(each_list)

        mask[torch.tensor(reactant_idx_list)] = 0.0
        reactant_logit = reactant_logit + mask
        reactant_hard = gumbel_softmax(reactant_logit, hard=True)
        reactant_soft = gumbel_softmax(reactant_logit,
                                       **self.gumbel_softmax_kwargs)
        reactant_idx = torch.where(reactant_hard == 1)[0].item()
        if hard:
            return template_idx, reactant_idx
        return template_soft.to(device), reactant_soft.to(device)

    def fit(self,
            episode_memory,
            env,
            n_epochs=1000,
            n_inner_epochs=10,
            batch_size=10,
            actor_update_freq=10,
            episode_gen_freq=1,
            incremental_episode_size=1,
            gumbel_softmax_annealing=0.95,
            target_act_kwargs={'std': 0.2, 'clip': 0.5},
            optimizer='Adagrad',
            optimizer_kwargs={'lr': 3e-4},
            bc_weight=0.,
            burnin_period=0,
            print_freq=1,
            logger=print,
            **kwargs):
        ''' train the actor using `env`

        Parameters
        ----------
        bc_weight : float
            weight for behavior cloning
        '''
        tau_init = self.gumbel_softmax_kwargs['tau']
        action_space = env.action_space
        for each_step in range(self.max_step+1):
            self.critic_1_list[each_step].init_optimizer(optimizer, optimizer_kwargs)
            self.critic_2_list[each_step].init_optimizer(optimizer, optimizer_kwargs)
        self.init_optimizer(optimizer, optimizer_kwargs)

        torch.manual_seed(self.gen_seed())
        critic_1_loss = None
        critic_2_loss = None
        actor_loss = None
        template_selector_loss = None

        self.critic_1_list[self.max_step] = CriticLast(env, self.device)
        self.critic_2_list[self.max_step] = CriticLast(env, self.device)

        for each_epoch in range(n_epochs):
            for each_step in range(self.max_step-1, -1, -1):
                self.gumbel_softmax_kwargs['tau'] = tau_init
                if each_epoch % episode_gen_freq == 0 and episode_gen_freq != -1:
                    gen_episode_memory = self.gen_episode_memory(env, incremental_episode_size, shutdown_everytime=True)
                    episode_memory.append(gen_episode_memory)
                for each_inner_epoch in range(n_inner_epochs * each_epoch, n_inner_epochs * (each_epoch+1)):
                    self.gumbel_softmax_kwargs['tau'] = tau_init * (gumbel_softmax_annealing ** each_inner_epoch)
                    minibatch_transition_list = episode_memory.sample_minibatch(batch_size, each_step)
                    state_action_train_list, y_train \
                        = self.construct_critic_dataset(minibatch_transition_list,
                                                        action_space,
                                                        target_act_kwargs)
                    critic_1_loss = self.critic_1_list[each_step].fit(
                        state_action_train_list,
                        y_train,
                        action_space,
                        n_epochs=1)
                    critic_2_loss = self.critic_2_list[each_step].fit(
                        state_action_train_list,
                        y_train,
                        action_space,
                        n_epochs=1)
                    if each_inner_epoch % actor_update_freq == 0 and each_inner_epoch > burnin_period:
                        if each_step != self.max_step:
                            actor_loss = self.actor_fit(minibatch_transition_list,
                                                        each_step,
                                                        action_space,
                                                        bc_weight,
                                                        n_epochs=1)
                            #template_selector_loss = self.template_selector_fit(minibatch_transition_list,
                            #                                                    each_step,
                            #                                                    action_space,
                            #                                                    n_epochs=1)
                    if each_inner_epoch % print_freq == 0:
                        logger(' iter: {}'.format(each_inner_epoch))
                        logger(' * critic_loss : {}, {}'.format(critic_1_loss, critic_2_loss))
                        logger(' * actor_loss : {}'.format(actor_loss))
                        logger(' * template_selector_loss : {}'.format(template_selector_loss))
                        logger(' * gs temperature : {}'.format(self.gumbel_softmax_kwargs['tau']))
        self.critic_1_list[self.max_step] = NullModule(self.device)
        self.critic_2_list[self.max_step] = NullModule(self.device)


    def actor_fit(self,
                  transition_list,
                  n_step,
                  action_space,
                  bc_weight,
                  n_epochs=10):
        for _ in range(n_epochs):
            state_action_list = [(each_transition.state,
                                  self.soft_act(each_transition.state, action_space))
                                 for each_transition in transition_list]
            loss = -self.critic_1_list[n_step].batch_forward(state_action_list, action_space).mean()
            pred_true_action_list = [(each_transition.action,
                                      self.soft_act(each_transition.state, action_space))
                                     for each_transition in transition_list\
                                     if each_transition.action[0] is not None]
            template_true_array = torch.tensor([each_pred_true_tuple[0][0]
                                                for each_pred_true_tuple in pred_true_action_list],
                                               device=self.device)
            template_pred_array = torch.stack([each_pred_true_tuple[1][0]
                                               for each_pred_true_tuple in pred_true_action_list])
            reactant_true_array = torch.tensor([each_pred_true_tuple[0][1]
                                                for each_pred_true_tuple in pred_true_action_list],
                                               device=self.device)
            reactant_pred_array = torch.stack([each_pred_true_tuple[1][1]
                                               for each_pred_true_tuple in pred_true_action_list])
            loss = loss + 0.5 * bc_weight * self.template_selector_loss(template_pred_array, template_true_array)\
                + 0.5 * bc_weight * self.template_selector_loss(reactant_pred_array, reactant_true_array)
            self.template_selector_optimizer_list[n_step].zero_grad()
            self.reactant_selector_optimizer_list[n_step].zero_grad()
            loss.backward()
            self.template_selector_optimizer_list[n_step].step()
            self.reactant_selector_optimizer_list[n_step].step()
        return loss.item()

    def template_selector_fit(self,
                              transition_list,
                              n_step,
                              action_space,
                              n_epochs=10):
        feature_list = [self.obs2feature(observation=each_transition.state)
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
        feature_tensor = torch.stack(feature_list)
        for _ in range(n_epochs):
            tgt_list = [torch.where(each_transition.action[0] == 1.0)[0]
                        for each_transition in transition_list
                        if each_transition.action != action_space.stop_action]
            pred_logit = self.template_selector_list[n_step].batch_forward(feature_tensor)
            tgt_tensor = torch.cat(tgt_list).to(self.device)
            loss = self.template_selector_loss(pred_logit, tgt_tensor)
            self.template_selector_optimizer_list[n_step].zero_grad()
            loss.backward()
            self.template_selector_optimizer_list[n_step].step()
        return loss.item()

    def construct_critic_dataset(self,
                                 transition_list,
                                 action_space,
                                 target_act_kwargs):
        state_action_train_list = []
        reward_list = []
        next_state_action_train_list = []
        y_train_list = []
        for each_transition in transition_list:
            state_action_train_list.append((each_transition.state,
                                            each_transition.action))
            if each_transition.state['n_step'] == self.max_step:
                next_action = action_space.stop_action
            else:
                with torch.no_grad():
                    next_action = self.target_act(each_transition.next_state,
                                                  action_space,
                                                  **target_act_kwargs)
            next_state_action_train_list.append((each_transition.next_state,
                                                 next_action))
            reward_list.append(each_transition.reward)
        with torch.no_grad():
            if each_transition.state['n_step'] == self.max_step:
                target_q = 0.
            else:
                target_q = torch.min(
                    torch.stack([self.critic_1_list[each_transition.state['n_step']+1].batch_forward(
                        next_state_action_train_list,
                        action_space),
                                 self.critic_2_list[each_transition.state['n_step']+1].batch_forward(
                                     next_state_action_train_list,
                                     action_space)]),
                    dim=0)[0]
            y_train = torch.tensor(reward_list, device=self.device) + self.discount * target_q
        return state_action_train_list, y_train

