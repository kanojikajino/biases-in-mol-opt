''' utilities
'''

from abc import abstractmethod
from collections import OrderedDict
import importlib
from numpy.random import default_rng
import luigi
import numpy as np
import torch


MAX_INT = 2 ** 31 - 1

class RngMixin:

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            raise ValueError('please call `set_seed` first.')
        return self._rng

    def set_seed(self, seed):
        self._rng = default_rng(seed)

    def seed_list(self, n_seed):
        return self.rng.integers(
            MAX_INT,
            size=n_seed)

    def gen_seed(self):
        return int(self.rng.integers(
            MAX_INT,
            size=1))

    def seeding_iter(self, param_dict, n_iter, non_seeding_param_dict=None):
        def _substitute_seed(param_dict):
            out_param_dict = dict()
            for each_key, each_val in param_dict.items():
                if isinstance(each_val, (dict,
                                         OrderedDict,
                                         luigi.freezing.FrozenOrderedDict)):
                    out_param_dict[each_key] = _substitute_seed(each_val)
                elif each_key == 'seed':
                    out_param_dict[each_key] = self.gen_seed()
                else:
                    out_param_dict[each_key] = each_val
            return out_param_dict

        param_dict_list = []
        for _ in range(n_iter):
            generated_param_dict = _substitute_seed(param_dict)
            if non_seeding_param_dict is not None:
                generated_param_dict.update(non_seeding_param_dict)
            param_dict_list.append(generated_param_dict)
        return param_dict_list


def delete_rng(input_obj):
    if hasattr(input_obj, '_torch_rng'):
        input_obj._torch_rng_state = input_obj._torch_rng.get_state()
        del input_obj._torch_rng
    if hasattr(input_obj, '_torch_rng_cpu'):
        input_obj._torch_rng_cpu_state = input_obj._torch_rng_cpu.get_state()
        del input_obj._torch_rng_cpu
    try:
        for each_attr in input_obj.__dict__.keys():
            delete_rng(getattr(input_obj, each_attr))
    except:
        pass


class TorchRngMixin:

    @property
    def torch_rng(self):
        if not hasattr(self, '_torch_rng'):
            self._torch_rng = torch.Generator(device=self.device if hasattr(self, 'device') else 'cpu')
            if hasattr(self, '_torch_rng_state'):
                self._torch_rng.set_state(self._torch_rng_state)
                del self._torch_rng_state
            else:
                try:
                    self._torch_rng.manual_seed(self._seed)
                except:
                    raise ValueError('please call `set_torch_seed` first to set seed.')
        return self._torch_rng

    @property
    def torch_rng_cpu(self):
        if not hasattr(self, '_torch_rng_cpu'):
            self._torch_rng_cpu = torch.Generator(device='cpu')
            if hasattr(self, '_torch_rng_cpu_state'):
                self._torch_rng_cpu.set_state(self._torch_rng_cpu_state)
                del self._torch_rng_cpu_state
            else:
                self._torch_rng_cpu.manual_seed(
                    int(torch.randint(
                        high=MAX_INT,
                        size=(1,),
                        generator=self.torch_rng,
                        device=self.device if hasattr(self, 'device') else 'cpu').to('cpu')))
        return self._torch_rng_cpu
        
    def set_torch_seed(self, seed):
        if hasattr(self, '_torch_rng'):
            self.torch_rng.manual_seed(int(seed))
            self.torch_rng_cpu.manual_seed(
                int(torch.randint(
                    high=MAX_INT,
                    size=(1,),
                    generator=self.torch_rng,
                    device=self.device if hasattr(self, 'device') else 'cpu').to('cpu')))
        else:
            self._seed = int(seed)

    def gen_seed(self):
        return int(torch.randint(high=MAX_INT, size=(1,), generator=self.torch_rng_cpu))

    def delete_rng(self):
        delete_rng(self)



class OptimizerMixin:

    def init_optimizer(self,
                       optimizer='Adagrad',
                       optimizer_kwargs={'lr': 1e-2}):
        self.optimizer = getattr(torch.optim, optimizer)(
            params=self.parameters(),
            **optimizer_kwargs)

    def _fit(self,
             X,
             y,
             weight=None,
             batch_size=1,
             n_epochs=100,
             weight_decay=1.0,
             max_update=None,
             optimizer='Adagrad',
             optimizer_kwargs={'lr': 1e-2},
             print_freq=10,
             logger=print):
        assert len(X) == len(y)
        sample_size = len(X)
        if weight is not None:
            weight = weight ** weight_decay
        optimizer = getattr(torch.optim, optimizer)(
            params=self.parameters(), **optimizer_kwargs)

        if max_update is not None:
            n_epochs = math.ceil(batch_size * max_update / sample_size)
            logger(' * n_epochs is changed to {}'.format(n_epochs))
        else:
            max_update = n_epochs * sample_size
        # training
        for iter_idx in range(max_update):
            running_loss = 0
            counter = 0
            idx_tensor = torch.randperm(sample_size,
                                        generator=self.torch_rng_cpu)[:batch_size]
            each_X = X[idx_tensor]
            each_y = y[idx_tensor]
            if weight is not None:
                each_weight = weight[idx_tensor]
            else:
                each_weight = None
            loss = self.loss(each_X,
                             each_y,
                             weight=each_weight,
                             with_regularizer=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter += 1
            if print_freq > 0 and iter_idx % print_freq == 0:
                logger('#(update) = {}\t loss = {}'.format(
                    iter_idx,
                    running_loss/counter))
        return running_loss/counter

    @abstractmethod
    def loss(self, X, y, weight=None, with_regularizer=False):
        raise NotImplementedError


class DeviceContext(torch.cuda.device):

    def __init__(self, device):
        if torch.cuda.is_available():
            super().__init__(device)
        else:
            pass

    def __enter__(self):
        if torch.cuda.is_available():
            super().__enter__()
        else:
            pass

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            super().__exit__(type, value, traceback)
        else:
            pass

def device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1

def device_name(device_idx):
    if torch.cuda.is_available():
        return 'cuda:{}'.format(device_idx % device_count())
    else:
        return 'cpu'


def shape_checker(array_like, shape):
    np_array = np.array(array_like)
    if np_array.shape != shape:
        raise ValueError('{} is inconsistent with {}'.format(array_like.shape, shape))
    return np_array


def class_importer(class_name):
    module_name, class_name = class_name.rsplit('.', 1)
    return getattr(importlib.import_module(module_name), class_name)


def get_instance(dict_params, **kwargs):
    kwargs.update(dict_params['kwargs'])
    return class_importer(dict_params['name'])(**kwargs)
