import torch
import covshift.weight
import molgym.chem.fingerprint


class FingerprintWeightEstimator:
    
    def __init__(self, fingerprint_params, weight_estimator_params, device='cpu'):
        self.device = device
        self.fingerprint = getattr(
            molgym.chem.fingerprint,
            fingerprint_params['module'])(
                **fingerprint_params['kwargs'])
        self.weight_estimator_params = weight_estimator_params

    def fit(self, src_mol_list, tgt_mol_list, src_tgt_list=None, logger=print, **fit_kwargs):
        if hasattr(self.fingerprint, 'fit'):
            self.fingerprint.fit(src_mol_list, src_tgt_list, logger=logger, **fit_kwargs)
        X_src = self.fingerprint.batch_forward(src_mol_list)
        X_tgt = self.fingerprint.batch_forward(tgt_mol_list)

        lengthscale = torch.norm(X_src, dim=1).mean()
        density_ratio_estimator = getattr(covshift.weight,
                                          self.weight_estimator_params['module'])(
                                              lengthscale=lengthscale,
                                              **self.weight_estimator_params['kwargs'])
        return torch.Tensor(density_ratio_estimator.fit(X_src, X_tgt))
