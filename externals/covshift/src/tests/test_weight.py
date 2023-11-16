''' test weight
'''

import unittest
import numpy as np
from sklearn.linear_model import RidgeCV
from covshift.utils import RngMixin
from covshift.weight import KernelMeanMatching, KernelULSIF


class TestKernelMeanMatching(unittest.TestCase, RngMixin):

    def setUp(self):
        self.ratio_estimator = KernelMeanMatching('RBFKernel',
                                                  {},
                                                  10.,
                                                  1e-3,
                                                  seed=43,
                                                  lengthscale=0.5)

    def test_fit(self):
        self.set_seed(46)
        w_true = np.array([1.0])
        X_train = self.rng.normal(0., 1., 100).reshape(100, 1)
        X_test = self.rng.normal(1., 1., 100).reshape(100, 1)
        y_train = X_train @ w_true
        y_test = X_test @ w_true

        beta_train = self.ratio_estimator.fit(X_train, X_test)

        clf_wo_covshift = RidgeCV()
        clf_wo_covshift.fit(X_train, y_train)
        y_pred_wo_covshift = clf_wo_covshift.predict(X_test)

        clf_w_covshift = RidgeCV()
        clf_w_covshift.fit(X_train, y_train, beta_train)
        y_pred_w_covshift = clf_w_covshift.predict(X_test)
        loss_wo_covshift = np.linalg.norm(y_test - y_pred_wo_covshift)
        loss_w_covshift = np.linalg.norm(y_test - y_pred_w_covshift)

        print(' * loss w/o covshift\t= {}'.format(loss_wo_covshift))
        print(' * loss w/ covshift\t= {}'.format(loss_w_covshift))
        self.assertLess(loss_w_covshift, loss_wo_covshift)

class TestKernelULSIF(TestKernelMeanMatching):

    def setUp(self):
        self.ratio_estimator = KernelULSIF('RBFKernel',
                                           {},
                                           1e-6,
                                           seed=43,
                                           lengthscale=0.5)
