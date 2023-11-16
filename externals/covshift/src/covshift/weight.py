''' calculate weights for covariate shift
'''

from abc import abstractmethod, ABCMeta
from cvxopt import matrix, solvers
import covshift.kernels
import numpy as np
import torch
from sklearn.model_selection import KFold
from mlbasics.utils import RngMixin


class WeightEstimatorBase(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X_src, X_tgt):
        pass


class KernelWeightEstimatorBase(WeightEstimatorBase, RngMixin):

    def compute_ker(self, X_src, X_tgt, tgt_tgt=False):
        with torch.no_grad():
            if isinstance(X_src, np.ndarray):
                X_src = torch.DoubleTensor(X_src)
                X_tgt = torch.DoubleTensor(X_tgt)
            elif isinstance(X_src, torch.Tensor):
                X_src = X_src.to(device='cpu', dtype=torch.float64)
                X_tgt = X_tgt.to(device='cpu', dtype=torch.float64)
            else:
                raise ValueError
            ker_src_src = self.kernel_obj.forward(X_src, X_src).numpy()
            ker_src_tgt = self.kernel_obj.forward(X_src, X_tgt).numpy()
            if tgt_tgt:
                ker_tgt_tgt = self.kernel_obj.forward(X_tgt, X_tgt).numpy()
                return ker_src_src, ker_src_tgt, ker_tgt_tgt
        return ker_src_src, ker_src_tgt

    def compute_train_test_ker(self, X_src_train, X_src_test, X_tgt_train, X_tgt_test):
        with torch.no_grad():
            if isinstance(X_src_train, np.ndarray):
                X_src_train = torch.DoubleTensor(X_src_train)
                X_src_test = torch.DoubleTensor(X_src_test)
                X_tgt_train = torch.DoubleTensor(X_tgt_train)
                X_tgt_test = torch.DoubleTensor(X_tgt_test)
            elif isinstance(X_src_train, torch.Tensor):
                X_src_train = X_src_train.to(device='cpu', dtype=torch.float64)
                X_src_test = X_src_test.to(device='cpu', dtype=torch.float64)
                X_tgt_train = X_tgt_train.to(device='cpu', dtype=torch.float64)
                X_tgt_test = X_tgt_test.to(device='cpu', dtype=torch.float64)
            else:
                raise ValueError
            ker_src_test_src_train = self.kernel_obj.forward(X_src_test, X_src_train).numpy()
            ker_src_test_tgt_train = self.kernel_obj.forward(X_src_test, X_tgt_train).numpy()
            ker_tgt_test_src_train = self.kernel_obj.forward(X_tgt_test, X_src_train).numpy()
            ker_tgt_test_tgt_train = self.kernel_obj.forward(X_tgt_test, X_tgt_train).numpy()
        return (ker_src_test_src_train,
                ker_src_test_tgt_train,
                ker_tgt_test_src_train,
                ker_tgt_test_tgt_train)

    def sampling(self, X_src, X_tgt):
        if len(X_src) > self.max_sample_size:
            X_src = torch.tensor(self.rng.choice(
                X_src,
                self.max_sample_size,
                replace=False)).to(X_src.device)
        if len(X_tgt) > self.max_sample_size:
            X_tgt = torch.tensor(self.rng.choice(
                X_tgt,
                self.max_sample_size,
                replace=False)).to(X_tgt.device)
        return X_src, X_tgt


class KernelMeanMatching(KernelWeightEstimatorBase):

    ''' calculate beta(x) = p_tgt(x) / p_src(x) by matching kernel mean embeddings

    Attributes
    ----------
    kernel_name : str
        name of kernel in gpytorch
    kernel_kwargs : dict
        parameters of the kernel
    max_beta : float
        
    '''

    def __init__(self,
                 kernel_name,
                 kernel_kwargs,
                 max_beta,
                 tol,
                 seed,
                 lengthscale=None,
                 max_sample_size=float('inf')):
        self.set_seed(seed)
        self.kernel_obj = getattr(covshift.kernels, kernel_name)(**kernel_kwargs)
        if max_beta <= 0:
            raise ValueError('max_beta must be positive')
        if tol <= 0:
            raise ValueError('tol must be positive')
        self.max_beta = max_beta
        self.tol = tol
        if hasattr(self.kernel_obj, 'lengthscale') and lengthscale is not None:
            self.kernel_obj.lengthscale = torch.Tensor([lengthscale])
        self.max_sample_size = max_sample_size

    def fit(self, X_src, X_tgt=[]):
        ''' compute the importance ratio that should be used for likelihood weighting.

        Parameters
        ----------
        X_src : array, (n_src, dim)
         X_tgt : array, (n_tgt, dim)

        Returns
        -------
        beta : array, (n_src,)
            p_tgt(x) / p_src(x)
        '''
        if len(X_tgt) == 0:
            return np.ones(X_src.shape[0])

        X_src, X_tgt = self.sampling(X_src, X_tgt)

        ker_src_src, ker_src_tgt = self.compute_ker(X_src, X_tgt)
        n_src = len(X_src)
        n_tgt = len(X_tgt)

        P = matrix(ker_src_src)
        q = matrix(-0.5 * ker_src_tgt @ np.ones(n_tgt)/n_tgt)
        G = matrix(np.vstack([np.identity(n_src),
                              -np.identity(n_src),
                              -np.ones((1, n_src)),
                              np.ones((1, n_src))]))
        h = matrix(np.vstack([self.max_beta * np.ones((n_src, 1)),
                              np.zeros((n_src, 1)),
                              -1.0 + self.tol,
                              1.0 + self.tol]))
        sol = solvers.qp(P, q, G, h)

        return n_src * np.array(sol['x']).ravel()


class KernelULSIF(KernelWeightEstimatorBase):

    def __init__(self,
                 kernel_name,
                 kernel_kwargs,
                 lmbd_list,
                 seed,
                 lengthscale=None,
                 n_splits=5,
                 max_sample_size=float('inf')):
        self.set_seed(seed)
        self.kernel_obj = getattr(covshift.kernels, kernel_name)(**kernel_kwargs)
        self.n_splits = n_splits
        if not hasattr(lmbd_list, '__iter__'):
            lmbd_list = [lmbd_list]
        for each_lmbd in lmbd_list:
            if each_lmbd < 0:
                raise ValueError('lmbd must be non-negative')
        self.lmbd_list = lmbd_list
        if self.kernel_obj.has_lengthscale and lengthscale is not None:
            self.kernel_obj.lengthscale = torch.Tensor([lengthscale])
            self.lengthscale_list = [lengthscale]
            for each_exponent in range(1, 6):
                self.lengthscale_list.append(lengthscale * (2 ** each_exponent))
                self.lengthscale_list.append(lengthscale * (2 ** (-each_exponent)))
        else:
            self.lengthscale_list = [None]

        self.max_sample_size = max_sample_size

    @property
    def lengthscale(self):
        if self.kernel_obj.has_lengthscale:
            return self.kernel_obj.lengthscale
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, lengthscale):
        if self.kernel_obj.has_lengthscale and lengthscale is not None:
            self.kernel_obj.lengthscale = torch.Tensor([lengthscale])

    def fit(self, X_src, X_tgt=[]):
        if len(X_tgt) == 0:
            return np.ones(X_src.shape[0])

        cv_score_list = []
        hyp_list = []

        X_src, X_tgt = self.sampling(X_src, X_tgt)

        kf = KFold(n_splits=self.n_splits)
        split_src = list(kf.split(X_src))
        split_tgt = list(kf.split(X_tgt))

        ker_src_src, ker_src_tgt, ker_tgt_tgt = self.compute_ker(X_src, X_tgt, tgt_tgt=True)
        for each_lmbd in self.lmbd_list:
            for each_lengthscale in self.lengthscale_list:
                self.lmbd = each_lmbd
                self.lengthscale = each_lengthscale
                cv_score = 0.
                for each_fold_idx in range(self.n_splits):
                    train_src_idx, test_src_idx = split_src[each_fold_idx]
                    train_tgt_idx, test_tgt_idx = split_tgt[each_fold_idx]
                    ker_src_train_src_train = ker_src_src[train_src_idx, :][:, train_src_idx]
                    ker_src_train_tgt_train = ker_src_tgt[train_src_idx, :][:, train_tgt_idx]
                    ker_src_test_src_train = ker_src_src[test_src_idx, :][:, train_src_idx]
                    ker_src_test_tgt_train = ker_src_tgt[test_src_idx, :][:, train_tgt_idx]
                    ker_tgt_test_src_train = ker_src_tgt[train_src_idx, :][:, test_tgt_idx].transpose()
                    ker_tgt_test_tgt_train = ker_tgt_tgt[test_tgt_idx, :][:, train_tgt_idx]
                    cv_score += self.obj_func(ker_src_train_src_train,
                                              ker_src_train_tgt_train,
                                              ker_src_test_src_train,
                                              ker_src_test_tgt_train,
                                              ker_tgt_test_src_train,
                                              ker_tgt_test_tgt_train)
                cv_score = cv_score / self.n_splits
                cv_score_list.append(cv_score)
                hyp_list.append((each_lmbd, each_lengthscale))
        best_idx = np.argmin(cv_score_list)
        best_hyp = hyp_list[best_idx]
        self.lmbd, self.lengthscale = best_hyp
        print(' * best lmbd, lengthscale = {}, {}'.format(self.lmbd, self.lengthscale))
        return self._fit(ker_src_src, ker_src_tgt)


    def _fit(self, ker_src_src, ker_src_tgt):
        ''' compute the importance ratio that should be used for likelihood weighting.

        Parameters
        ----------
        X_src : array, (n_src, dim)
        X_tgt : array, (n_tgt, dim)

        Returns
        -------
        beta : array, (n_src,)
            p_tgt(x) / p_src(x)
        '''
        n_src, n_tgt = ker_src_tgt.shape
        alpha = self.compute_alpha(ker_src_src, ker_src_tgt)
        return np.clip(ker_src_src @ alpha + (1.0 / (n_tgt * self.lmbd)) * ker_src_tgt.sum(axis=1), a_min=0., a_max=None)

    def compute_alpha(self, ker_src_src, ker_src_tgt):
        n_src, n_tgt = ker_src_tgt.shape
        from numpy.linalg import solve, cond
        A = (1.0/n_src) * ker_src_src + self.lmbd * np.identity(n_src)
        b = - (1.0/(n_src * n_tgt * self.lmbd)) * ker_src_tgt.sum(axis=1)
        alpha = solve(A, b)
        return alpha

    def obj_func(self,
                 ker_src_train_src_train,
                 ker_src_train_tgt_train,
                 ker_src_test_src_train,
                 ker_src_test_tgt_train,
                 ker_tgt_test_src_train,
                 ker_tgt_test_tgt_train):
        alpha_train = self.compute_alpha(ker_src_train_src_train, ker_src_train_tgt_train)
        len_tgt_train = ker_src_train_tgt_train.shape[1]
        w_src_test = ker_src_test_src_train @ alpha_train + (1.0 / (len_tgt_train * self.lmbd)) * ker_src_test_tgt_train.sum(axis=1)
        w_tgt_test = ker_tgt_test_src_train @ alpha_train + (1.0 / (len_tgt_train * self.lmbd)) * ker_tgt_test_tgt_train.sum(axis=1)
        return 0.5 * (w_src_test ** 2).mean() - w_tgt_test.mean()
