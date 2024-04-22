import torch
import torch.nn as nn

from ....learning.model.common.pointnet import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointPickPolicyModel(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    """

    def __init__(self, dim=3, c_dim=3, hidden_dim=128, out_dim=1):
        """
        Args:
            dim (int): input points dimension (xyzrgb)
            c_dim (int): dimension of incoming conditioning code c. Currently we use RGB.
            hidden_dim (int): hidden dimension of the network
            out_dim (int): out dimension
        """
        super().__init__()

        self.c_dim = c_dim

        # If condition is given or not
        if c_dim == 0:
            self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, hidden_dim)
            self.fc_c = nn.Linear(c_dim, hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, out_dim)

        self.actvn = nn.LeakyReLU(0.2)
        self.pool = maxpool


    def forward(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            p: Input point cloud. (B, num_points, dim)
            c: Conditioning code. (B, c_dim)

        Returns:
            torch.Tensor: 
                Pointwise classification logits. 
                Please make sure to take the softmax activation after this.
        """
        batch_size, T, D = p.size()

        # output size: B x T X F

        # If condition is given or not...
        if self.c_dim == 0:
            net = self.fc_pos(p)
            net = self.block_0(net)
        else:
            net_input = self.fc_pos(p)
            net_c = self.fc_c(c)
            net = torch.cat([net_input, net_c], dim=2)
            net = self.block_0(net)
        # PointNet layer 0
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)                           # Values are not activated here.

        # PointNet layer 1
        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # PointNet layer 2        
        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # PointNet layer 3
        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # PointNet layer (Output) 4
        net = self.block_4(net)
        net = net.squeeze(-1)

        # Output logit value
        return net
    

if __name__=="__main__":

    model = PointPickPolicyModel(dim=3, c_dim=0)
    temp_input = torch.rand((2, 1024, 3))
    out = model(temp_input, None)

    print(out.shape)
