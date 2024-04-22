import torch
import torch.nn as nn


from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.common.cvae import CVAE


class HistoryPlacePolicyonly(nn.Module):

    def __init__(self, cvae_config, fetching_gpt_config):
        super().__init__()
        self.cvae = CVAE(cvae_config)
        self.fetching_gpt = GPT2FetchingConditioner(fetching_gpt_config)


    def forward(self, x, *args, **kargs):
        """forward
        (For training)

        Args:
            x (th.Tensor): Label
            *args, **kargs (th.Tensor): Input to the GPT.

        Returns:
            recon_x (th.Tensor): Reconstructed `x`
            mean (th.Tensor): mean
            log_var (th.Tensor): log var
        """
        
        c = self.fetching_gpt(*args, **kargs)
        recon_x, mean, log_var = self.cvae(x, c)

        return recon_x, mean, log_var
    

    def inference(self, *args, **kargs) -> torch.Tensor:
        """For inference
        
        Args:
            *args, **kargs (th.Tensor): Input to the GPT

        Returns:
            recon_x (th.Tensor): Reconstructed `x`
        """
        c = self.fetching_gpt(*args, **kargs)
        recon_x = self.cvae.inference(c)

        return recon_x
    