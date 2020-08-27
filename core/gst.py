# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):

    def __init__(self, idim=80, ref_enc_filters=[32, 32, 64, 64, 128, 128], ref_dim=128, is_text=False):

        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i])
             for i in range(K)])

        self.is_text = is_text
        out_channels = self.calculate_channels(idim, 3, 2, 1, K)

        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_dim,
                          batch_first=True)
        self.n_mel_channels = idim

    def forward(self, inputs):

        if self.is_text:
            out = inputs.unsqueeze(1)
        else:
            out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()

        _, out = self.gru(out)

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):

    def __init__(self, ref_dim=128, num_heads=4, token_num=10, token_dim=128):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, token_dim // num_heads))
        d_q = ref_dim
        d_k = token_dim // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_dim,
            num_heads=num_heads)
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1,
                                                          -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)
        return style_embed


class MultiHeadAttention(nn.Module):

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]

        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, hp, is_text=False):
        super().__init__()
        self.is_text = is_text
        if is_text:
            self.encoder_pre = ReferenceEncoder(idim=hp.encoder_embedding_dim, is_text=is_text)
            self.encoder_post = ReferenceEncoder(idim=hp.encoder_embedding_dim, is_text=is_text)
            self.stl = STL(ref_dim=hp.ref_enc_gru_size*2, num_heads=hp.num_heads, token_num=hp.token_num, \
                           token_dim=hp.context_embedding_size)
        else:
            self.encoder = ReferenceEncoder(idim=hp.n_mel_channels)
            self.stl = STL(ref_dim=hp.ref_enc_gru_size, num_heads=hp.num_heads, token_num=hp.token_num, \
                           token_dim=hp.acoustic_embedding_size)

    def forward(self, inputs, post_inputs=None):
        if self.is_text:
            pre_context_embed = self.encoder_pre(inputs)
            post_context_embed = self.encoder_post(post_inputs)
            context_embed = torch.cat((pre_context_embed, post_context_embed), dim=1)
        else:
            context_embed = self.encoder(inputs)
        style_embed = self.stl(context_embed)
        return style_embed