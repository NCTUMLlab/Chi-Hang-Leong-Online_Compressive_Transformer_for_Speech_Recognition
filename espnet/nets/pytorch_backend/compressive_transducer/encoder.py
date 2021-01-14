#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.compressive_transducer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import (PositionalEncoding,ScaledPositionalEncoding)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
    PositionwiseFeedForwardGLU
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsamplingNoPosition
)


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
        feedforward_GLU=False,
        compressive_rate=3,
        memspeech_size=10
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if input_layer == "conv2d":
            self.embed = Conv2dSubsamplingNoPosition(idim, attention_dim, dropout_rate)
            self.position = PositionalEncoding(attention_dim, dropout_rate)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        self.memspeech_size = memspeech_size
        self.compressive_rate = compressive_rate
        if positionwise_layer_type == "linear":
            if feedforward_GLU:
                positionwise_layer = PositionwiseFeedForwardGLU
            else:
                positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        else:
            raise NotImplementedError("Support only linear.")
        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                torch.nn.Conv1d(in_channels=attention_dim,out_channels=attention_dim,
                                kernel_size=compressive_rate,stride=compressive_rate),
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks, memsh=None, memsh_mask=None,is_inference=False, is_compress=True):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, (Conv2dSubsampling, Conv2dSubsamplingNoPosition)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        new_mem = []
        loss = None
        if memsh is not None:
            masks_kv = torch.cat((memsh_mask[:,:,-self.memspeech_size:],masks),dim=2) if memsh_mask is not None else None
            xs_q = self.position(xs)
            xs_kv = None
            for mem,en in zip(memsh, self.encoders):
                if xs_kv is None: # means first layer
                    xs_kv = torch.cat((mem[:,-self.memspeech_size:],xs),dim=1)    # xs  mem[:,-self.memspeech_size:].detach()
                    xs_kv = self.position(xs_kv)
                else:
                    xs_kv = torch.cat((mem[:,-self.memspeech_size:],xs_q),dim=1)    # xs_q mem[:,-self.memspeech_size:].detach()

                xs_q, _, new_mem_loss = en(xs_q, masks, xs_kv, masks_kv, is_inference, is_compress)
                if is_compress:
                    new_mem_i, loss_i = new_mem_loss
                    loss = loss + loss_i if loss is not None else loss_i
                    new_mem.append(torch.cat((mem,new_mem_i),dim=1))

            xs = xs_q
            if is_compress:
                new_mask = masks[:,:,::self.compressive_rate] #torch.ones(new_mem_i.size(0),1,new_mem_i.size(1)).type(masks.dtype).to(xs.device) 
                masks = torch.cat((memsh_mask,new_mask),dim=2) if memsh_mask is not None else None
        else:
            xs, masks, _ = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_mem, loss

    def update_commem(self, memsh):
        new_memsh = []
        for mem,en in zip(memsh, self.encoders):
            new_mem_i = en.update_commem()
            new_memsh.append(torch.cat((mem,new_mem_i),dim=1))

        return new_memsh

