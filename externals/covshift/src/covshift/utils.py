''' Utilities
'''

from numpy.random import default_rng

class RngMixin:

    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = default_rng()
        return self._rng

    def set_seed(self, seed=None):
        self._rng = default_rng(seed)
