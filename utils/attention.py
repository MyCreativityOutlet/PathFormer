"""
PathFormer
Copyright (C) 2026  Liam Brydon
Contact at: liam.brydon@auckland.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch.nn import Linear, Module, GELU, LayerNorm, Dropout, Sequential
from torch.nn.functional import scaled_dot_product_attention


class KVCacheCausalAttention(Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = Linear(d_model, d_model * 3)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None, cache=None, layer_idx=0, max_len=None):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        if cache is not None:
            # Step 0: Prefill Phase (Initialise static cache)
            if layer_idx not in cache:
                static_k = torch.zeros((B, self.n_heads, max_len, self.head_dim), device=x.device, dtype=x.dtype)
                static_v = torch.zeros((B, self.n_heads, max_len, self.head_dim), device=x.device, dtype=x.dtype)

                # Insert the prefill tokens
                static_k[:, :, :L, :] = k
                static_v[:, :, :L, :] = v

                cache[layer_idx] = (static_k, static_v)
                k, v = static_k[:, :, :L, :], static_v[:, :, :L, :]

            # Step > 0: Decode Phase (Insert in-place)
            else:
                static_k, static_v = cache[layer_idx]
                seq_len = cache["seq_len"]  # Current new token position for each item in the batch

                # Cast new tokens to the cache's dtype to prevent in-place assignment errors
                batch_idx = torch.arange(B, device=x.device)
                static_k[batch_idx, :, seq_len, :] = k.squeeze(2).to(static_k.dtype)
                static_v[batch_idx, :, seq_len, :] = v.squeeze(2).to(static_v.dtype)

                # Slice out the historical context up to the maximum current length
                max_seq_len = seq_len.max().item()
                k = static_k[:, :, :max_seq_len + 1, :].to(q.dtype)
                v = static_v[:, :, :max_seq_len + 1, :].to(q.dtype)

        is_causal = (cache is None and L > 1) and mask is None
        if mask is not None:
            mask = mask.logical_not()
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)


class KVCacheCrossAttention(Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Linear(d_model, d_model)
        self.kv_proj = Linear(d_model, d_model * 2)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, x, memory, memory_mask=None, cache=None, layer_idx=0):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Use a distinct cache key so it doesn't collide with self-attention caches
        cache_key = f"cross_layer_{layer_idx}"

        if cache is not None:
            # Step 0: Prefill Phase (Compute K, V once for the static context)
            if cache_key not in cache:
                kv = self.kv_proj(memory).chunk(2, dim=-1)
                k, v = [t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2) for t in kv]
                cache[cache_key] = (k, v)

            # Step > 0: Decode Phase (Retrieve static K, V)
            else:
                k, v = cache[cache_key]
        else:
            # No caching (e.g., standard training forward pass)
            kv = self.kv_proj(memory).chunk(2, dim=-1)
            k, v = [t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2) for t in kv]
        # 3. Attention
        if memory_mask is not None:
            memory_mask = memory_mask.logical_not()
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask=memory_mask, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)


class KVCacheDecoderLayer(Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, activation=None):
        super().__init__()
        if activation is None:
            activation = GELU()
        self.norm1 = LayerNorm(d_model)
        self.self_attn = KVCacheCausalAttention(d_model, n_heads, dropout)  # Reused from PathFormer
        self.self_attn_dropout = Dropout(dropout)
        self.cross_attn_dropout = Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = KVCacheCrossAttention(d_model, n_heads, dropout)
        self.norm3 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, dim_feedforward),
            activation,
            Dropout(dropout),
            Linear(dim_feedforward, d_model),
            Dropout(dropout)
        )

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, cache=None, layer_idx=0, max_len=None):
        x = x + self._sa_block(self.norm1(x), mask=tgt_mask, cache=cache, layer_idx=layer_idx, max_len=max_len)
        x = x + self._cross_attn_block(self.norm2(x), memory, memory_mask=memory_mask, cache=cache, layer_idx=layer_idx)
        x = x + self.ff(self.norm3(x))
        return x

    def _sa_block(self, x, mask=None, cache=None, layer_idx=0, max_len=None):
        x = self.self_attn(x, mask=mask, cache=cache, layer_idx=layer_idx, max_len=max_len)
        return self.self_attn_dropout(x)

    def _cross_attn_block(self, x, memory, memory_mask=None, cache=None, layer_idx=0):
        x = self.cross_attn(x, memory, memory_mask=memory_mask, cache=cache, layer_idx=layer_idx)
        return self.cross_attn_dropout(x)
