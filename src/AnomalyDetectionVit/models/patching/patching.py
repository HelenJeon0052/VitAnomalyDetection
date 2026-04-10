import torch
import torch.nn as nn




class ViT3DPatchEmbed(nn.Module):
    """
    transforms 3D medical volumes into sequences of tokens
    u = f(x, y, z)
    """

    def __init__(self, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        print("patch_size _ init:", self.patch_size)
        self.proj = nn.Conv3d(in_channels=in_channels, out_channels=embed_dim, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        # [B, dim, Dd, Ll, Ww]
        """print("input.shape:", x.shape)
        print("input.dtype:", x.dtype)
        print("patch_size _ forward:", self.patch_size)"""
        x = self.proj(x)
        # print("input.dtype:", x.shape)
        B, C, Dd, Ll, Ww = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, T, C]
        # print("input.dtype:", x.shape)

        return x, (Dd, Ll, Ww)

class PatchMerging3D(nn.Module):
    """
    Downsample by 2
    Increase channels
    stride = 2 
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        # feat : [B, C, D, L, W]
        feat = self.conv(feat)
        print(f'feat.shape: {feat.shape}') # [B, out, D/2, L/2, W/2]
        B, C, D, L, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        return tokens, (D, L, W), feat