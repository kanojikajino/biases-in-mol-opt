from gpytorch.kernels import *
import torch


class TanimotoSimilarity(Kernel):

    def forward(self, x1, x2):
        kernel = torch.zeros(x1.shape[0],
                             x2.shape[0],
                             dtype=torch.float64)
        for each_row_idx, each_row in enumerate(x1):
            numerator = (each_row * x2).sum(axis=1)
            denominator = torch.clip(each_row + x2, max=1.).sum(axis=1)
            kernel[each_row_idx, :] = numerator / denominator
        return kernel
