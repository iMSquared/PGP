import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


from learning.model.common.transformer import GPT2FetchingConditioner

class ValueNet(nn.Module):

    @dataclass
    class Config:
        dim_condition: int


    def __init__(self, config: Config):
        super(ValueNet, self).__init__()

        self.config = config
        self.fc1 = nn.Linear(config.dim_condition, config.dim_condition)
        self.fc2 = nn.Linear(config.dim_condition, 1)

    def forward(self, latent_condition):
        accumulated_reward = F.relu(self.fc1(latent_condition))
        accumulated_reward = self.fc2(accumulated_reward)

        return accumulated_reward



class HistoryPlaceValueonly(nn.Module):

    def __init__(self, fetching_gpt_config, value_config):
        super().__init__()   
        self.fetching_gpt = GPT2FetchingConditioner(fetching_gpt_config)
        self.value_net = ValueNet(value_config)


    def forward(self, *args, **kargs):
        """forward
        (For training)

        To forward the value only, do the following:
        ```
        c = model.fetching_gpt(*args, **kargs)
        value = model.value_net(c)
        ```
        
        Args:
            *args, **kargs (th.Tensor): Input to the GPT.

        Returns:
            value (th.Tensor): Predicted value
        """
        
        c = self.fetching_gpt(*args, **kargs)
        value = self.value_net(c)

        return value
    
    
    def inference_value_only(self, *args, **kargs) -> torch.Tensor:
        """For inference of value only.

        Args:
            *args, **kargs (th.Tensor): Input to the GPT

        Returns:
            value (th.Tensor): Predicted value
        """
        c = self.fetching_gpt(*args, **kargs)
        value = self.value_net(c)

        return value
