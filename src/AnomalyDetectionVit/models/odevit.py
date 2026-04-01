import torch
import torch.nn as nn



from AnomalyDetectionVit.models.splitting import SplitODEBlock
from AnomalyDetectionVit.attention import AttentionMLP, MLP, FrictionField

class ODETokenField(nn.Module):
    """
    d/dx(u) = -P(u) ⊙ u + Q(u) ⊙ Block(u)
    P : friction / retention gate
    Q : forcing / injection gate
    """

    def __init__(self, dim, num_heads=6, dropout=0.0, mlp_ratio=3.0, attn_dropout=0.0):
        super().__init__()
        self.block = AttentionMLP(dim, num_heads=num_heads, dropout=dropout, mlp_ratio=mlp_ratio, attn_dropout=attn_dropout)
        self.p_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh())
        self.q_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh())

    def forward(self, t, u):
        # u : [B, T, D]
        P = self.p_gate(u)
        Q = self.q_gate(u)

        forcing = self.block(u)
        du = -P*u + Q*forcing

        return du

dim = 384
attn_field = AttentionMLP(dim, num_heads=6, dropout=0.0, mlp_ratio=3.0, attn_dropout=0.0)
mlp_field = MLP(dim, dropout=0.0, mlp_ratio=3.0)
fric_field = FrictionField(dim)


token_mixer = SplitODEBlock(
    attn_field=attn_field,
    mlp_field=mlp_field,
    fric_field=fric_field,
    mode='strang',
    steps_attn=6,
    steps_mlp=2,
    steps_fric=1,
    use_friction=True,
    friction_position='mid'
)
x = torch.randn(1, 10, dim)
x_out = token_mixer(x) # x : [B, T+1, dim]