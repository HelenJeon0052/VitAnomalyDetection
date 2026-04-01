import torch
import torch.nn as nn
import torch.nn.functional as F

from AnomalyDetectionVit.models.mlps import GatedSkip, LightMLPRefine

class MLPDecoder(nn.Module):
    """
    features: [f1, f2, f3, f4] (low to high)
    decode : high to low and output logits
    """

    def __init__(self, embed_dim_rev, num_classes, dropout=0.0):
        super().__init__()
        # embed_dim_rev = [c4, c3, c2, c1]
        c4, c3, c2, c1 = embed_dim_rev

        self.up43 = GatedSkip(in_main=c4, in_skip=c3, out=c3)
        self.ref43 = LightMLPRefine(c3, mlp_ratio=2.0, dropout=dropout)

        self.up32 = GatedSkip(in_main=c3, in_skip=c2, out=c2)
        self.ref32 = LightMLPRefine(c2, mlp_ratio=2.0, dropout=dropout)

        self.up21 = GatedSkip(in_main=c2, in_skip=c1, out=c1)
        self.ref21 = LightMLPRefine(c1, mlp_ratio=2.0, dropout=dropout)

        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)
    def forward(self, feats):
        f1, f2, f3, f4 = feats

        x = f4
        x = F.interpolate(x, size=f3.shape[-3:], mode='trilinear', align_corners=False)
        x = self.up43(x, f3)
        x = x + self.ref43(x)

        x = F.interpolate(x, size=f2.shape[-3:], mode='trilinear', align_corners=False)
        x = self.up32(x, f2)
        x = x + self.ref32(x)

        x = F.interpolate(x, size=f1.shape[-3:], mode='trilinear', align_corners=False)
        x = self.up21(x, f1)
        x = x + self.ref21(x)

        logits = self.head(x)

        return logits

