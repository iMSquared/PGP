import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, NamedTuple
from dataclasses import dataclass, field
from simple_parsing import Serializable


class ObservationCNN(nn.Module):
    """WTF"""

    def __init__(self, dim_input: int, dim_output: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_input, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(2048, dim_output)

    def forward(self, x):
        y = self.conv1(x)   # -> (B, 16, 62, 62)
        y = self.relu1(y)
        y = self.pool1(y)   # -> (B, 16, 31, 31)
        y = self.conv2(y)   # -> (B, 16, 29, 29)
        y = self.relu2(y)
        y = self.pool2(y)   # -> (B, 32, 14, 14)
        y = self.conv3(y)   # -> (B, 64, 12, 12)
        y = self.relu3(y)
        y = self.pool3(y)   # -> (B, 64, 6, 6)
        y = self.conv4(y)
        y = self.relu4(y)   # -> (B, 128, 4, 4)
        
        y = y.reshape(y.shape[0], -1) # -> (B, 2048)
        y = self.fc1(y)

        return y


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, max_len  : int,
                       dim_embed: int):
        super().__init__()

        pe = th.zeros(max_len, dim_embed)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        
        div_term = th.exp(th.arange(0, dim_embed, 2).float() * (-math.log(10000.0) / dim_embed))
        
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x  



class Value(nn.Module):

    def __init__(self, dim_hidden: int, 
                       dim_embed : int):
        """Value in self-attention

        Args:
            dim_hidden (int): d_v(=d_k?) in the paper
            dim_embed (int): Word embedding dim
        """
        super().__init__()
        # Bias=false makes it matrix.
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False) 


    def forward(self, x):
        x = self.fc1(x)
        return x



class Key(nn.Module):

    def __init__(self, dim_hidden: int, 
                       dim_embed : int):
        """Key in self-attention

        Args:
            dim_hidden (int): d_k in the paper
            dim_embed (int): Word embedding dim
        """
        super().__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)


    def forward(self, x):
        x = self.fc1(x)
        return x



class Query(nn.Module):

    def __init__(self, dim_hidden: int, 
                       dim_embed : int):
        """Query in self-attention

        Args:
            dim_hidden (int): d_k in the paper
            dim_embed (int): Word embedding dim
        """
        super().__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
    

    def forward(self, x):
        x = self.fc1(x)
        return x



class Attention(nn.Module):

    def __init__(self, dropout_rate=0.1):
        """Attention layer

        Args:
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, Q, K, V, attn_mask=None):
        """
        Attention(Q, K, V) = norm(QK)V

        Args:
            Q (torch.Tensor): Q tensor
            K (torch.Tensor): K tensor
            V (torch.Tensor): V tensor
            attn_mask (torch.Tensor, optional): Mask

        Returns:
            attn_v (torch.Tensor): Weighted values, shape=(batch_size, num_q_seq, num_hidden)
            attn_p (torch.Tensor): Attention score, shape=(batch_size, num_q_seq, num_k_seq)
        """

        # Attention score: shape=(batch_size, num_q_seq, num_k_seq)
        a = th.matmul(Q, K.transpose(-1,-2).float())
        a /= th.sqrt(th.tensor(Q.shape[-1]).float()) # Scaled

        # Mask (optional)
        if attn_mask is not None:
            a.masked_fill_(attn_mask, -1e9)          # Negative infinity

        # Softmax along the last dim. 
        attn_p = th.softmax(a, -1)                   

        # Weighted values
        attn_v = th.matmul(self.dropout(attn_p), V)  # (num_q_seq, dim_hidden)

        return attn_v, attn_p
    


class MultiHeadAttention(th.nn.Module):

    def __init__(self, dim_hidden  : int,
                       dim_head    : int,
                       num_heads   : int,
                       dropout_rate: float = 0.1):
        """Multi-head attention layer

        Args:
            dim_hidden (int): Hidden dimension of multi-head output
            dim_head (int): Hidden dimension of each head
            num_heads (int): Number of heads
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_head   = dim_head
        self.num_heads  = num_heads

        self.W_Q             = Query(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_K             = Key(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_V             = Value(self.dim_hidden, self.dim_head * self.num_heads)
        self.scaled_dot_attn = Attention(dropout_rate)
        self.fc1             = nn.Linear(self.dim_head * self.num_heads, self.dim_hidden)
        self.dropout         = nn.Dropout(dropout_rate)
    

    def forward(self, x, attn_mask=None):
        """forward

        Args:
            x (th.Tensor): shape=(batch_size, dim_head*num_heads)
            attn_mask (th.Tensor, optional): Attension mask, shape=(??). Defaults to None.

        Returns:
            output (th.Tensor): Multi-head weighted values, shape=(batch_size, num_q_seq, dim_hidden)
            attn_p (th.Tensor): Attention scores of all heads, shape=(batch_size, num_heads, num_q_seq, num_k_seq)
        """
        batch_size = x.shape[0]

        # Calculate `num_heads` number of heads at once.
        # (batch_size, num_heads, num_q_seq, dim_head)
        q_s = self.W_Q(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        k_s = self.W_K(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        v_s = self.W_V(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        # Mask (optional): shape=(batch_size, num_heads, num_q_seq, num_k_seq)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)#.transpose(0,1)
        
        # attn_v: (batch_size, num_heads, num_q_seq, dim_head)
        # attn_p: (batch_size, num_heads, num_q_seq, num_k_seq)
        attn_v, attn_p = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)

        # Calculate MultiHead
        # (batch_size, num_q_seq, num_heads, dim_heads)
        attn_v = attn_v.transpose(1, 2).contiguous()
        # (batch_size, num_q_seq, num_heads*dim_head)
        attn_v = attn_v.view(batch_size, -1, self.num_heads*self.dim_head)
        # (batch_size, num_q_seq, dim_hidden)
        output = self.fc1(attn_v)
        output = self.dropout(output)

        return output, attn_p
    


class FeedForwardNetwork(nn.Module):

    def __init__(self, dim_hidden: int,
                       dim_ffn   : int):
        """Feed forward network of the transformer
        This is a word-wise simple MLP operation.
        Conv1D is used for the implementation.

        Args:
            dim_hidden (int): Hidden dimension
            dim_ffn (int): Internal dimension of the feed-forward network
        """
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_ffn    = dim_ffn

        self.conv1  = nn.Conv1d(in_channels  = self.dim_hidden, 
                                out_channels = self.dim_ffn,
                                kernel_size  = 1)               # Word-wise operation
        self.conv2  = nn.Conv1d(in_channels  = self.dim_ffn, 
                                out_channels = self.dim_hidden,
                                kernel_size  = 1)               # Word-wise operation
        self.act_fn = F.gelu # original: ReLU


    def forward(self, inputs):
        """forward

        Args:
            inputs (th.Tensor): Normalized multi-head weighted values, shape=(batch_size, num_q_seq, dim_hidden)

        Returns:
            th.Tensor: FFN(x)
        """
        # (batch_size, dim_ffn, num_seq)
        output = self.act_fn(self.conv1(inputs.transpose(1, 2)))
        # (batch_size, num_seq, dim_hidden)
        output = self.conv2(output).transpose(1, 2)

        return output



class GPT2DecoderLayer(nn.Module):

    def __init__(self, dim_hidden  : int,
                       dim_head    : int,
                       num_heads   : int,
                       dim_ffn     : int,
                       dropout_rate: float=0.1): 
        """GPT2 decoder layer (based on the transformer)

        Args:
            dim_hidden (int): Hidden dimension
            dim_head (int): Hidden dimension of each head
            num_heads (int): Number of heads
            dim_ffn (int): Internal dimension of the feed-forward network
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.self_attn   = MultiHeadAttention(dim_hidden, dim_head, num_heads, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(dim_hidden)
        self.ffn         = FeedForwardNetwork(dim_hidden, dim_ffn)
        self.layer_norm2 = nn.LayerNorm(dim_hidden)
        self.dropout     = nn.Dropout(dropout_rate)
    

    def forward(self, x, attn_mask):
        """forward
        GPT consists of decoder transformer block only. 
        - Keep masked self-attention and FFN.
        - Remove decoder-encoder self-attention.

        Args:
            x (th.Tensor): Input, shape=(batch_size, num_dec_seq, dim_hidden)
            attn_mask (th.Tensor): Decoder mask for attention

        Returns:
            ffn_out: Decoder output
            self_attn_prob: Decoder attention score
        """

        # Self multi-head attention + normalization
        # self_attn_out:  (batch_size, num_dec_seq, dim_hidden)
        # self_attn_prob: (batch_size, num_heads, num_dec_seq, num_dec_seq)
        self_attn_out, self_attn_prob = self.self_attn(x, attn_mask)
        self_attn_out = self.layer_norm1(x + self.dropout(self_attn_out))

        # Feed-forward layer + normalization
        # (batch_size, num_dec_seq, dim_hidden)
        ffn_out = self.ffn(self_attn_out)
        ffn_out = self.layer_norm2(self_attn_out + self.dropout(ffn_out))

        return ffn_out, self_attn_prob



class GPT2FetchingConditioner(nn.Module):

    @dataclass
    class Config(Serializable):
        """Default configuration"""
        # Data type
        image_res          : int
        dim_obs_rgbd_ch    : int
        dim_obs_rgbd_encode: int
        dim_obs_grasp      : int
        dim_action_input   : int    
        dim_goal_input     : int      
        # Embedding
        # dim_embed       : int           # Same with dim_hidden
        # Architecture
        dim_hidden      : int
        num_heads       : int           # dim_head * num_heads should match with dim_hidden.
        dim_ffn         : int
        num_gpt_layers  : int
        dropout_rate    : float
        # Position encoding
        max_len         : int           # Positional encoding period
        seq_len         : int           
        # Output
        dim_condition   : int

        @property
        def dim_head(self) -> int:
            return int(self.dim_hidden / self.num_heads)

        @property
        def dim_embed(self) -> int:
            return self.dim_hidden


    def __init__(self, config: Config):
        """GPT2FetchingPlace

        Args:
            config (Config): configuration
        """
        super().__init__()
        self.config = config

        # Embedding layers (Reward is directly embedded)
        self.encode_obs_rgbd = ObservationCNN(config.dim_obs_rgbd_ch, config.dim_obs_rgbd_encode)
        self.embed_obs       = nn.Linear(config.dim_obs_rgbd_encode+config.dim_obs_grasp,
                                         config.dim_embed)
        self.embed_action    = nn.Linear(config.dim_action_input, 
                                         config.dim_embed)
        # Positional encoding
        self.pos_embed       = PositionalEncoding(max_len = config.max_len,
                                                  dim_embed = config.dim_embed)
        
        # Common layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ln      = nn.LayerNorm(config.dim_embed)

        # Transformer blocks, |NOTE(jiyong)| https://michigusa-nlp.tistory.com/26
        self.layers = []
        for _ in range(config.num_gpt_layers):
            self.layers.append(GPT2DecoderLayer(dim_hidden   = config.dim_hidden,
                                                dim_head     = config.dim_head,
                                                num_heads    = config.num_heads,
                                                dim_ffn      = config.dim_ffn,
                                                dropout_rate = config.dropout_rate))
        self.layers = nn.ModuleList(self.layers)

        # Output layer
        self.fc_condi = nn.Linear(config.dim_hidden, config.dim_condition)


    def forward(self, init_obs_rgbd  : th.Tensor,
                      init_obs_grasp : th.Tensor,
                      goal           : th.Tensor,
                      seq_action     : th.Tensor,
                      seq_obs_rgbd   : th.Tensor,
                      seq_obs_grasp  : th.Tensor,
                      mask_seq_action: th.Tensor,
                      mask_seq_obs   : th.Tensor,) -> th.Tensor: 
        """forward

        Args:
            init_obs_rgbd (th.Tensor): Initial rgbd observation, `shape=(batch_size, dim_obs_rgbd_ch, res, res)`
            init_obs_grasp (th.Tensor): Initial grasp observation, `shape=(batch_size, 1)
            goal (th.Tensor): Goal condition token, `shape=(batch_size, dim_goal_input)`
            seq_action (th.Tensor): Sequence of actions, `shape=(batch_size, seq_len, dim_action_input)`
            seq_obs_rgbd (th.Tensor): Sequence of rgbd observations, `shape=(batch_size, seq_len, dim_obs_rgbd_ch, res, res)`
            seq_obs_grasp (th.Tensor): Sequence of grasp observations, `shape=(batch_size, 1)`
            mask_seq_action (th.Tensor): Sequence mask, `shape=(batch_size, seq_len)`
            mask_seq_obs (th.Tensor): Sequence mask, `shape=(batch_size, seq_len)`

        Returns:
            th.Tensor: Predicted conditioning vector `shape=(batch_size, dim_condition)
        """
        batch_size = seq_action.shape[0]

        # Embedding start_token (init_obs)
        init_obs_rgbd_encoded = self.encode_obs_rgbd(init_obs_rgbd)
        init_obs = th.cat((init_obs_rgbd_encoded, init_obs_grasp), dim=-1)
        start_token = self.embed_obs(init_obs)

        # Embedding action sequenece
        seq_action_token = self.embed_action(seq_action)

        # Embedding observation sequence
        #   Embed observation using batch trick
        seq_obs_rgbd = seq_obs_rgbd.view(-1, self.config.dim_obs_rgbd_ch, self.config.image_res, self.config.image_res)
        seq_obs_rgbd_encoded = self.encode_obs_rgbd(seq_obs_rgbd)
        seq_obs_rgbd_encoded = seq_obs_rgbd_encoded.view(batch_size, self.config.seq_len, self.config.dim_obs_rgbd_encode)
        seq_obs = th.cat((seq_obs_rgbd_encoded, seq_obs_grasp), dim=-1)
        seq_obs_token = self.embed_obs(seq_obs)

        # Activate and add pos embeding 
        # (pos embedding should be shared among same time step)
        start_token = F.gelu(start_token)
        seq_action_token = self.pos_embed(F.gelu(seq_action_token))
        seq_obs_token    = self.pos_embed(F.gelu(seq_obs_token))


        # Combine embedding (NOTE(ssh): This is a little confusing...)
        #   (a, a, a), (o, o, o) -> (a, o), (a, o), (a, o)
        seq_token = th.stack((seq_action_token, seq_obs_token), dim=-2)
        seq_mask  = th.stack((mask_seq_action, mask_seq_obs), dim=-1)
        #   (a, o), (a, o), (a, o) -> (a, o, a, o, a, o)
        seq_token = seq_token.flatten(start_dim=-3, end_dim=-2)     # BxSx2xZ -> Bx2SxZ
        seq_mask  = seq_mask.flatten(start_dim=-2, end_dim=-1)
        input_seq = th.cat((start_token.unsqueeze(dim=-2), seq_token), dim=-2)

        # Append 1 to the mask for the start stoken.
        mask = th.cat((th.ones((batch_size, 1)).int().to(seq_mask.device), seq_mask), dim=-1)
        #   Encoder mask
        pad_mask = mask.unsqueeze(1)
        pad_mask = pad_mask.repeat(1, self.config.seq_len*2+1, 1).bool()
        #   Reform to decoder(causal) mask
        lookahead_mask = th.tril(th.ones_like(pad_mask).bool(), diagonal=0)  
        lookahead_mask = ~lookahead_mask                      # (B, 2*seq+1, 2*seq+1), 

        # Residual dropout (see paper)
        input_seq = self.dropout(input_seq)
        input_seq = self.ln(input_seq)

        # Masked self-attention blocks
        dec_outputs, attn_prob = self.layers[0](input_seq, lookahead_mask)
        for l, layer in enumerate(self.layers[1:]):
            dec_outputs, attn_prob = layer(dec_outputs, lookahead_mask)

        # Get predictions
        seq_last_index = th.sum(mask.int(), dim = -1).unsqueeze(-1)-1
        batch_indices  = th.arange(dec_outputs.size(0)).unsqueeze(-1)
        last_dec_outputs = dec_outputs[batch_indices, seq_last_index].squeeze(-2) # Select the last hidden dim for all batches.
        out = F.gelu(self.fc_condi(last_dec_outputs))

        return out  # (batch_size, dim_condition)
    