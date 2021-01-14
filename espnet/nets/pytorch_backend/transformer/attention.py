#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        if(len(query.shape)==3):
            n_batch = query.size(0)
            q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
            q = q.transpose(1, 2)  # (batch, head, time1, d_k)
            k = k.transpose(1, 2)  # (batch, head, time2, d_k)
            v = v.transpose(1, 2)  # (batch, head, time2, d_k)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, head, time1, time2)
            if mask is not None:
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
                min_value = float(
                    numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
                )
                scores = scores.masked_fill(mask, min_value)
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

            p_attn = self.dropout(self.attn)
            x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
            x = (
                x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            )  # (batch, time1, d_model)
        else:
            n_batch = query.size(0)
            n_chunk = query.size(1)
            q = self.linear_q(query).view(n_batch, n_chunk, -1, self.h, self.d_k)
            k = self.linear_k(key).view(n_batch, n_chunk, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, n_chunk, -1, self.h, self.d_k)
            q = q.transpose(2, 3)  # (batch, chunk, head, time1, d_k)
            k = k.transpose(2, 3)  # (batch, chunk, head, time2, d_k)
            v = v.transpose(2, 3)  # (batch, chunk, head, time2, d_k)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, chunk, head, time1, time2)
            if mask is not None:
                mask = mask.unsqueeze(2).eq(0)  # (batch, chunk, 1, time1, time2)
                min_value = float(
                    numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
                )
                scores = scores.masked_fill(mask, min_value)    #error
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, chunk, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, chunk, head, time1, time2)

            p_attn = self.dropout(self.attn)
            x = torch.matmul(p_attn, v)  # (batch, chunk, head, time1, d_k)
            x = (
                x.transpose(2, 3).contiguous().view(n_batch, n_chunk, -1, self.h * self.d_k)
            )  # (batch, chunk, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def simple_attention(self,query,key,value):
        n_batch = query.size(0)
        q = torch.nn.functional.linear(query,self.linear_q.weight) #,self.linear_q.bias.detach()) don't use bias
        k = torch.nn.functional.linear(key,self.linear_k.weight)   #,self.linear_k.bias.detach())
        v = torch.nn.functional.linear(value,self.linear_v.weight) #,self.linear_v.bias.detach())
        scores = torch.matmul(q,k.transpose(-2,-1))             #/math.sqrt(self.d_k)
        attn = torch.softmax(scores,dim=-1)
        return torch.matmul(attn,v)

class MultiHeadedAttention4dim(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention4dim object."""
        super(MultiHeadedAttention4dim, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, chunk, time1, size)
        :param torch.Tensor key: (batch, chunk, time2, size)
        :param torch.Tensor value: (batch, chunk, time2, size)
        :param torch.Tensor mask: (batch, chunk, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, chunk, time1, d_model)
             weighted by the query dot key attention (batch, chunk, head, time1, time2)
        """
        n_batch = query.size(0)
        n_chunk = query.size(1)
        q = self.linear_q(query).view(n_batch, n_chunk, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, n_chunk, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, n_chunk, -1, self.h, self.d_k)
        q = q.transpose(2, 3)  # (batch, chunk, head, time1, d_k)
        k = k.transpose(2, 3)  # (batch, chunk, head, time2, d_k)
        v = v.transpose(2, 3)  # (batch, chunk, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, chunk, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(2).eq(0)  # (batch, chunk, 1, time1, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)    #error
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, chunk, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, chunk, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, chunk, head, time1, d_k)
        x = (
            x.transpose(2, 3).contiguous().view(n_batch, n_chunk, -1, self.h * self.d_k)
        )  # (batch, chunk, time1, d_model)
        return self.linear_out(x)  # (batch, chunk, time1, d_model)
        
    def simple_attention(self,query,key,value):
        n_batch = query.size(0)
        q = torch.nn.functional.linear(query,self.linear_q.weight) #,self.linear_q.bias.detach()) don't use bias
        k = torch.nn.functional.linear(key,self.linear_k.weight)   #,self.linear_k.bias.detach())
        v = torch.nn.functional.linear(value,self.linear_v.weight) #,self.linear_v.bias.detach())
        scores = torch.matmul(q,k.transpose(-2,-1))             #/math.sqrt(self.d_k)
        attn = torch.softmax(scores,dim=-1)
        return torch.matmul(attn,v)
