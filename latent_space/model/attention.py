import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from einops import rearrange, repeat
import math
import random


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = True,
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        # pdb.set_trace()
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TypeEncoder(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        win_size: int = 32,
        type: int = 3,
        d_model: int = 512,
        batch_first: bool = True,
        length: int = 34,
    ):
        super().__init__()
        self.win_size = win_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        pos_1 = torch.zeros(win_size).unsqueeze(1)
        pos_2 = torch.arange(type).unsqueeze(1)
        pos = torch.cat((pos_1, pos_2[1:]), 0)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        # pdb.set_trace()
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        d_x, atten = self.attention(self.ln_1(x))
        x = x + d_x
        x = x + self.mlp(self.ln_2(x))
        return x, atten


class KalmanAttentionBlock(nn.Module):
    """
    x -> [dim_x, batch, ensemble]
    y -> [dim_y, batch, ensemble]
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, xy: torch.Tensor, attn_mask):
        K = xy
        Q = x
        attn_mask = (
            attn_mask.to(dtype=x.dtype, device=x.device)
            if attn_mask is not None
            else None
        )
        return self.attn(Q, K, K, need_weights=True, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, xy: torch.Tensor, attn_mask):
        d_x, atten = self.attention(self.ln_1(x), self.ln_1(xy), attn_mask)
        x = x + d_x
        x = x + self.mlp(self.ln_2(x))
        return x, atten
