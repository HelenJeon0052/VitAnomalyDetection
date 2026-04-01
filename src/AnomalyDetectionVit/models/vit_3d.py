import torch
import torch.nn as nn


from AnomalyDetectionVit.models.decoder import MLPDecoder
from AnomalyDetectionVit.models.encoder import HierarchicalEncoder3D







"""
Residual Connections
feature map u : Represents the hidden state (the processed 3D features) at "time" (depth) t
P(x) : act as the weight matrix || friction > how much of the previous state is retained
Q(x) : represents the external input || forcing function from specific 3D spatial features of the medical scan
"""

class NeuralODEBlock(nn.Module):
    """
    u' + P(x)u = Q(x)
    u' = f(u, t)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x(t+1) = x(t) + \int f(x(t)) dt
        return self.lin(torch.relu(x))

class Light3DVit(nn.Module):
    def __init__(self,
                 in_channels=4,
                 num_classes=4,
                 embed_dim=(48, 96, 192), # (48, 96, 192, 384)
                 depths=(2, 2, 2), # (2, 2, 2, 2)
                 sr_ratios=(2, 1, 1), # (4, 2, 1, 1)
                 block_type='sr',
                 ode_mode='strang',
                 ode_steps_attn=2,
                 ode_steps_mlp=1,
                 ode_steps_fric=1,
                 use_friction=True,
                 friction_position='mid',
                 patch_size=4,
                 triage_pool='cls',
                 triage_num=256):
        super().__init__()

        self.encoder = HierarchicalEncoder3D(
            in_channels=in_channels,
            embed_dim=list(embed_dim),
            depth=list(depths),
            sr_ratio=list(sr_ratios),
            mlp_ratio=4.0,
            dropout=0.0,
            attn_drop=0.0,
            block_type=block_type,
            ode_mode=ode_mode,
            ode_steps_attn=ode_steps_attn,
            ode_steps_mlp=ode_steps_mlp,
            ode_steps_fric=ode_steps_fric,
            use_friction=use_friction,
            friction_position=friction_position,
            patch_size=patch_size
        )
        

        c4= embed_dim[-1]

        self.triage_pool = triage_pool
        self.triage_head = nn.Sequential(
            nn.Linear(c4, triage_num),
            nn.GELU(),
            nn.Linear(triage_num, 1)
        )

    

    def _pool3d(self, f4: torch.Tensor) ->  torch.Tensor:
        print(f"shapes in _pool3d: {f4}")
        
        if isinstance(f4, (list, tuple)):
            raise TypeError(f"Expected a single tensor for f4, but got {type(f4)}")
        
        if not torch.is_tensor(f4):
            raise TypeError(f"Expected a tensor for f4, got {type(f4)}")
        
        if f4.ndim != 5:
            raise ValueError(f"Expected f4 to have 5 dimensions (B, C, D, H, W), but got {f4.ndim} dimensions")
        
        if self.triage_pool == 'gap':
            return f4.mean(dim=(2, 3, 4))
        if self.triage_pool == 'gmp':
            return f4.amax(dim=(2, 3, 4))
        raise ValueError(f'Unknown triage_pool: {self.triage_pool}')

    def forward(self, x):
        feat_last, feats = self.encoder(x)

        pooled = self._pool3d(feats[-1])
        
        return self.triage_head(pooled)