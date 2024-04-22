import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from dataclasses import dataclass
from simple_parsing import Serializable



class CVAEEncoder(nn.Module):

    def __init__(self, encoder_layer_sizes: int,
                       latent_size        : int,
                       dim_condition      : int): 
        """I don't have time to doc this.

        Args:
            encoder_layer_sizes (_type_): _description_
            latent_size (_type_): _description_
            dim_condition (_type_): _description_
        """
        super().__init__()

        self.layer_sizes = list(encoder_layer_sizes)
        self.latent_size = latent_size
        self.dim_condition = dim_condition

        self.layer_sizes[0] += self.dim_condition

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(
                name="N{:d}".format(i), module=nn.LayerNorm(out_size)
            )
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1))

        self.linear_means = nn.Linear(self.layer_sizes[-1], self.latent_size)
        self.linear_log_var = nn.Linear(self.layer_sizes[-1], self.latent_size)


    def forward(self, x, c):
        # |TODO| check shape after cat
        x = th.cat((x, c), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class CVAEDecoder(nn.Module):

    def __init__(self, decoder_layer_sizes: Tuple[int],
                       latent_size        : int,
                       dim_condition      : int):
        super().__init__()
        self.layer_sizes = list(decoder_layer_sizes)
        self.latent_size = latent_size
        self.dim_condition = dim_condition

        input_size = self.latent_size + self.dim_condition

        self.MLP = nn.Sequential()


        for i, (in_size, out_size) in enumerate(zip([input_size]+self.layer_sizes[:-1], self.layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(self.layer_sizes):
                self.MLP.add_module(
                    name="N{:d}".format(i), module=nn.LayerNorm(out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1))


    def forward(self, z, c):
        z = th.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x



class CVAE(nn.Module):

    @dataclass
    class Config(Serializable):
        latent_size        : int
        dim_condition      : int
        dim_output         : int
        dim_embed          : int         
        encoder_layer_sizes: tuple
        decoder_layer_sizes: tuple


    def __init__(self, config: Config):
        """

        Args:
            config (Config, optional): _description_. Defaults to Config().
        """
        super().__init__()        
        self.config = config
        
        self.embed = nn.Linear(config.dim_output, config.dim_embed)
        self.encoder = CVAEEncoder(encoder_layer_sizes = config.encoder_layer_sizes,
                                   latent_size         = config.latent_size,
                                   dim_condition       = config.dim_condition)
        self.decoder = CVAEDecoder(decoder_layer_sizes = config.decoder_layer_sizes,
                                   latent_size         = config.latent_size,
                                   dim_condition       = config.dim_condition)


    def forward(self, x, c):
        """forward
        (For training)

        Args:
            x (th.Tensor): Label
            c (th.Tensor): Conditioning vector from other network.

        Returns:
            recon_x (th.Tensor): Reconstructed `x`
            mean (th.Tensor): mean
            log_var (th.Tensor): log var
        """

        x = F.leaky_relu(self.embed(x))
        c = c.squeeze()    # Squeeze any zero dim from the condition output.

        mean, log_var = self.encoder(x, c)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, mean, log_var


    def reparameterize(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)

        return mu + eps * std


    def inference(self, c):

        z: th.Tensor = th.randn([c.size(0), self.config.latent_size]).to(c.device)
        recon_x = self.decoder(z, c)

        return recon_x