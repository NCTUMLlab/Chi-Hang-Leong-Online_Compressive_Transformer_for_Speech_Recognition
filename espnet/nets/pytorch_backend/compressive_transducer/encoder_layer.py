#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        compressivesh,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.commemsh = compressivesh   #conv1d
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_q, mask_q, x_kv=None, mask_kv=None, is_inference=False, is_compress=True):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param Bool is_inference: if is inference, it will be True

        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x_q

        if self.normalize_before:
            x_q = self.norm1(x_q)
            if x_kv is not None:
                x_kv = self.norm1(x_kv)
        
        x_mem = x_q     #.detach()
        
        if x_kv is not None:
            x = x_kv
            mask = mask_kv
        else:
            x = x_q
            mask = mask_q

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if is_compress:  
            x_memsh = x_mem.transpose(-1,-2)
            x_memsh = self.commemsh(x_memsh)        #conv1d
            new_memsh = x_memsh.transpose(-1,-2)
            if not is_inference:
                x_mem_detach = x_mem.detach()    #use detach no grad
                prev_x = self.self_attn.simple_attention(x_mem_detach,x_mem_detach,x_mem_detach)
                new_x = self.self_attn.simple_attention(x_mem_detach,new_memsh,new_memsh)
                com_loss = torch.mean((new_x-prev_x)**2)
                new_mem = ( new_memsh , com_loss )
            else:
                new_mem = (new_memsh, torch.Tensor([0.])) # (mem, com_loss)
        else:
            new_mem = (None, None) # (mem, com_loss)
        self.x_mem = x_mem
            
        return x, mask, new_mem

    def update_commem(self):
        x_memsh = self.x_mem.transpose(-1,-2)
        x_memsh = self.commemsh(x_memsh)        #conv1d
        new_memsh = x_memsh.transpose(-1,-2)

        return new_memsh