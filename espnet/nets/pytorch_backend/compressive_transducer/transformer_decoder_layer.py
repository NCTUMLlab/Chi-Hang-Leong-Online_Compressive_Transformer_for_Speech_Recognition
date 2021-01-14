"""Decoder layer definition for transformer-transducer models."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module for transformer-transducer models.

    Args:
        size (int): input dim
        self_attn (MultiHeadedAttention): self attention module
        feed_forward (PositionwiseFeedForward): feed forward layer module
        dropout_rate (float): dropout rate
        normalize_before (bool): whether to use layer_norm before the first block
        concat_after (bool): whether to concat attention layer's input and output

    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)

        self.dropout = nn.Dropout(dropout_rate)

        self.size = size

        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1],
                tgt.shape[2] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1], tgt.shape[2]-1, self.size)}"
            tgt_q = tgt[:,:,-1:, :]
            residual = residual[:,:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:,:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
            
        self.x_q = x
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=2)

        return x, tgt_mask, memory, memory_mask

