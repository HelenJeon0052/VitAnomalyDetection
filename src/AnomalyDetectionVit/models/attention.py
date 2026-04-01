# --------------------------------------------------
# P(x) = friction / retention term : a gate for learning features
# Q(x) = forcing : Attention + MLP feature injection
# --------------------------------------------------




import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class AttentionField3D(nn.Module):
    def __init__(self, dim: int, num_heads: int=6, sr_ratio: float=1.0, dropout:float=0.0, attn_drop:float=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SRMultiheadAttention3D(dim, num_heads = num_heads, sr_ratio=sr_ratio, attn_drop = attn_drop, proj_drop = dropout)

    def forward(self, u: torch.Tensor, grid_shape: Tuple[int, int, int]):
        return self.attn(self.norm(u), grid_shape)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=3.5, drop=0.0):
        super().__init__()
        l = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, l)
        self.fc2 = nn.Linear(l, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x : [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2) # each [B, T, L, Ld]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, T, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        print(f'out.shape: {out.shape}')




        # projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class AttentionMLP(nn.Module):
    """
    a transformer block for discrete residual
    Residual behavior handled by the ODE integrator
    """
    def __init__(self, dim, num_heads=6, mlp_ratio=3.0, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim ,mlp_ratio, dropout)

    def forward(self, x, grid=None):
        """
        cf) residual
        x = x + attn(...)
        x = x + mlp(...)
        """
        attn_in = self.norm1(x)
        attn_out, attn_weights = self.attn(attn_in, attn_in, attn_in, need_weights=True)
        y = x + attn_out
        y = y + self.mlp(self.norm2(y))
        return y

class MLPField(nn.Module):
    def __init__(self, dim: int, mlp_ratio:float=3.0, dropout:float=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
    def forward(self, u: torch.Tensor):
        return self.mlp(self.norm(u))

class FrictionField(nn.Module):
    """
    du/dt = -P(u) * u (retention / contraction)
    P(u) (0, 1] via sigmoid gate
    """

    def __init__(self, dim:int):
        super().__init__()
        self.p_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, u:torch.Tensor):
        P = self.p_gate(u)
        return -P*u


class SRMultiheadAttention3D(nn.Module):
    """
    Spatial Reduction Attention for 3D tokens
     - Input tokens correspond to a 3D grid(D',L',W') > T = D'L'W'
     - Reduce K, V by a factor sr_ratio via Conv3d stride=sr_ratio
    """
    def __init__(self, dim, num_heads=6, sr_ratio=1.0, proj_drop=0.0, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim % num_heads == 0'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)

        if self.sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
            self.sr_norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, grid_shape):
        # x: [B, T, C], grid_shape = (Dd, Ll, Ww)
        B, T, C = x.shape
        Dd, Ll, Ww = grid_shape

        q = self.q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        print(f'q.shape: {q.shape}')

        if self.sr_ratio > 1:
            # reshape tokens back to 3D feature map
            feat = x.transpose(1, 2).reshape(B, C, Dd, Ll, Ww)
            feat = self.sr(feat)
            Ds, Ls, Ws = feat.shape[-3:]
            kv_in = feat.flatten(2).transpose(1, 2) # [B, Ts, C]
            kv_in = self.sr_norm(kv_in)
        else:
            kv_in = x
            Ds, Ls, Ws = Dd, Ll, Ww

        kv = self.kv(kv_in).reshape(B, kv_in.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # [B, L, T, Ld] > [B, C, D, L, W] > [B, C, D/s, L/s, W/s] > [B, Ts, C] > [B, Ts, L, Ld] (transpose: [B, L, Ts, Ld]) > [B, L, T, Ts] > [B, L, T, Ld] > [B, T, C]
        return out

class SRtransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads=6, sr_ratio=1, mlp_ratio=3.0, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRMultiheadAttention3D(dim, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim ,mlp_ratio, dropout)

    def forward(self, x, grid_shape):
        x = x + self.attn(self.norm1(x), grid_shape)
        x = x + self.mlp(self.norm2(x))

        return x
