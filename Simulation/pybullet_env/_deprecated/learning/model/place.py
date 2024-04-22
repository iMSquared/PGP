import pickle
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from simple_parsing import Serializable

import sys
sys.path.append('Simulation/pybullet_env/learning')
from learning.model.common.pointnet import ResnetBlockFC
from learning.model.common.cvae import CVAE


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PolicyModelPlaceBelief(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    """

    def __init__(self, config):
        """
        Args:
            dim (int): input points dimension (xyzrgbw)
            c_dim (int): dimension of incoming conditioning code c. Currently we use (rgbxy).
            hidden_dim (int): hidden dimension of the network
            out_dim (int): out dimension
        """
        super().__init__()

        self.config = config
        
        self.p_dim = config.dim_point
        self.c_dim = config.dim_goal
        self.hidden_dim = config.dim_pointnet
        self.c_hidden_dim = config.dim_goal_hidden
        self.out_dim = config.dim_action_place
        
        self.cvae = CVAE(config)

        # If condition is given or not
        if self.c_dim == 0:
            self.fc_p = nn.Linear(self.p_dim, self.hidden_dim + self.c_hidden_dim)
        else:
            self.fc_p = nn.Linear(self.p_dim, self.hidden_dim)
            self.fc_c = nn.Linear(self.c_dim, self.c_hidden_dim)
        self.block_0 = ResnetBlockFC(self.hidden_dim + self.c_hidden_dim, self.hidden_dim)
        self.block_1 = ResnetBlockFC(2 * self.hidden_dim, self.hidden_dim)
        self.block_2 = ResnetBlockFC(2 * self.hidden_dim, self.hidden_dim)
        self.block_3 = ResnetBlockFC(2 * self.hidden_dim, self.hidden_dim)

        self.actvn = nn.LeakyReLU(0.2)
        self.pool = maxpool


    def forward(self, p: th.Tensor, c: th.Tensor, a: th.Tensor) -> th.Tensor:
        """
        Args: 
            p: Input point cloud. (B, num_points, dim)
            c: Conditioning code. (B, c_dim)

        Returns:
         th.Tensor: 
                Pointwise classification logits. 
                Please make sure to take the softmax activation after this.
        """
        batch_size, T, D = p.size()

        # If condition is given or not...
        if self.c_dim == 0:
            net = self.fc_p(p)
            net = self.block_0(net)
        else:
            net_input = self.fc_p(p)
            net_c = self.fc_c(c).unsqueeze(1)
            net_c = net_c.expand(batch_size, T, -1)
            net = th.cat([net_input, net_c], dim=2)
            net = self.block_0(net)
        # PointNet layer 0
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)                           # Values are not activated here.

        # PointNet layer 1
        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)

        # PointNet layer 2        
        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)

        # PointNet layer 3
        net = self.block_3(net)
        b_enc_vec = self.pool(net, dim=1, keepdim=True)    # belief encoding vector size: B x self.hidden_dim

        # CVAE
        recon_x, mean, log_var, z = self.cvae(a, b_enc_vec)
        
        return recon_x, mean, log_var, z
    
    
    def inference(self, p: th.Tensor, c: th.Tensor):
        batch_size, T, D = p.size()

        # If condition is given or not...
        if self.c_dim == 0:
            net = self.fc_p(p)
            net = self.block_0(net)
        else:
            net_input = self.fc_p(p)
            net_c = self.fc_c(c)
            net_c = net_c.repeat(net_input.size(0), net_input.size(1), 1)
            net = th.cat([net_input, net_c], dim=2)
            net = self.block_0(net)
        # PointNet layer 0
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)                           # Values are not activated here.

        # PointNet layer 1
        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)

        # PointNet layer 2        
        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = th.cat([net, pooled], dim=2)

        # PointNet layer 3
        net = self.block_3(net)
        b_enc_vec = self.pool(net, dim=1, keepdim=True)    # belief encoding vector size: B x self.hidden_dim

        # CVAE
        recon_x = self.cvae.inference(b_enc_vec.squeeze(1))
        
        return recon_x



if __name__=="__main__":

    @dataclass
    class Setting(Serializable):
        device = 'cuda' if th.cuda.is_available() else 'cpu'
        # input
        dim_action_place = 3
        dim_action_embed = 16        
        dim_point = 7
        dim_goal = 5
        
        # PointNet
        dim_pointnet = 128
        dim_goal_hidden = 8

        # CVAE
        dim_vae_latent = 16
        dim_vae_condition = 16
        vae_encoder_layer_sizes = [dim_action_embed, dim_action_embed + dim_vae_condition, dim_vae_latent]
        vae_decoder_layer_sizes = [dim_vae_latent, dim_vae_latent + dim_vae_condition, dim_action_place]
    
    config = Setting()
    
    with open('dataset/sim_dataset_fixed_belief/2.25_0_99.pickle', 'rb') as f:
        data = pickle.load(f)
    
    p = th.tensor(data['belief'][2], dtype=th.float32).to(config.device)
    p = th.unsqueeze(p, 0)
    c = th.tensor(data['goal'], dtype=th.float32).to(config.device)
    c = th.unsqueeze(c, 0)
    target_action = th.tensor([data['action'][2][2][0], data['action'][2][2][1], data['action'][2][4]], dtype=th.float32).to(config.device)
    target_action = th.unsqueeze(target_action, 0)
    
    model = PolicyModelPlaceBelief(config).to(config.device)
    out_1 = model(p, c, target_action)
    out_2 = model.inference(p, c)
    
    print(out_1[0].shape, out_2.shape)
