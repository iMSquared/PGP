import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, NamedTuple
from dataclasses import dataclass

class ELBOLoss(nn.Module):

    @dataclass
    class LossData:
        recon: th.Tensor
        kld: th.Tensor
        beta: float
        
        @property
        def total(self) -> th.Tensor:
            return self.recon + self.beta*self.kld


    def __init__(self, beta: float, reduction="batchmean"):
        super(ELBOLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, recon_x, x, mean, log_var) -> LossData:
        loss = {}
        if self.reduction=="batchmean":
            recon_loss = F.mse_loss(recon_x, x)
            kld = -0.5 * th.sum( th.mean(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1) ) / x.size(0)
        elif self.reduction=="none":
            recon_loss = th.sum(F.mse_loss(recon_x, x, reduction="none"), dim=list(range(len(x.shape))[1:]))
            kld = -0.5 * th.mean(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)

        loss = self.LossData(recon=recon_loss, kld=kld, beta=self.beta)

        return loss