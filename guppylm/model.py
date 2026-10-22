"""
GuppyLM — a tiny fish brain.

Vanilla transformer: multi-head attention, ReLU FFN, LayerNorm, learned positional embeddings.
No GQA, no SwiGLU, no parallel residual, no RoPE. As simple as it gets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GuppyConfig


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if past_kv is not None:
            # KV cache: keys/values of earlier positions, shape (B, H, P, head_dim)
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))
        if use_cache:
            return out, (k, v)
        return out


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)
        self.down = nn.Linear(config.ffn_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        if use_cache:
            a, kv = self.attn(self.norm1(x), mask, past_kv, use_cache=True)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, kv
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GuppyLM(nn.Module):
    def __init__(self, config: GuppyConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    def forward_cached(self, idx, past_kvs=None):
        """Forward pass that also returns per-layer (k, v) for incremental decoding.

        idx holds only the NEW tokens; past_kvs holds the keys/values of the P tokens
        before them (or None for a prefill). Positions continue from P, so the result
        matches forward() on the full sequence as long as P + T <= max_seq_len.
        Returns (logits for the new tokens, list of (k, v) covering all P + T tokens).
        """
        B, T = idx.shape
        P = 0 if past_kvs is None else past_kvs[0][0].size(2)
        pos = torch.arange(P, P + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        # A single new token may attend to everything before it: no mask needed.
        mask = None
        if T > 1:
            mask = torch.ones(T, P + T, device=idx.device).tril(diagonal=P).unsqueeze(0).unsqueeze(0)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, mask, None if past_kvs is None else past_kvs[i], use_cache=True)
            new_kvs.append(kv)
        return self.lm_head(self.norm(x)), new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, use_cache=False, **kwargs):
        """Sample up to max_new_tokens. use_cache=True reuses past keys/values instead of
        re-running the whole prefix every step; it falls back to a full recompute once the
        sequence outgrows max_seq_len (the sliding window shifts every position)."""
        self.eval()
        past = None
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -self.config.max_seq_len:])
            elif past is not None and idx.size(1) <= self.config.max_seq_len:
                logits, past = self.forward_cached(idx[:, -1:], past)
            else:
                logits, past = self.forward_cached(idx[:, -self.config.max_seq_len:])
                if idx.size(1) >= self.config.max_seq_len:
                    past = None  # window is full: next step must recompute
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == self.config.eos_id:
                break
        return idx, []

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return total, 0

    def param_summary(self):
        total, _ = self.param_count()
        return f"GuppyLM: {total:,} params ({total/1e6:.1f}M)"
