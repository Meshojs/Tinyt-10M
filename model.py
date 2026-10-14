import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Module):
    """
    input x : (B , T)
    output x : (B , T , d_model)
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)  # [0, 2, 4, ...] = 2i
        div = 1.0 / (10000 ** (i / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.Wo(out)
        return out


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.rms_norm1 = nn.RMSNorm(d_model)  # dim=1
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.rms_norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mha(self.rms_norm1(x)))
        x = x + self.ffn(self.rms_norm2(x))

        return x


class TinyT(nn.Module):
    def __init__(self, vocab_size, d_model, n_blocks, seq_len, n_heads, d_ff, dropout):
        super().__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pe = PositionalEncoding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.rms_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(self, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def optim_config(self, weight_decay, betas, learning_rate):
        params_dict = {n: p for n, p in self.named_parameters()}
        params_dict = {n: p for n, p in params_dict.items() if p.requires_grad == True}
        decay_params = []
        nodecay_params = []

        for p in self.parameters():
            if not p.requires_grad:
                continue

            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            betas=betas,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
        )

        return optimizer

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(self.pe(x))
        for block in self.blocks:
            x = block(x)

        x = self.rms_norm(x)
        return self.head(x)
