import torch
import torch.nn as nn


from AnomalyDetectionVit.models.attention import AttentionMLP, SRtransformerBlock3D, AttentionField3D, MLPField, FrictionField
from AnomalyDetectionVit.models.splitting import SplitODEBlock
from AnomalyDetectionVit.models.patching.patching import ViT3DPatchEmbed, PatchMerging3D


class HierarchicalEncoder3D(nn.Module):
    """
    returns :
     - 3D feature maps
     - [f1, f2, f3, f4]
    """
    def __init__(self,
                 in_channels,
                 embed_dim,
                 depth,
                 sr_ratio,
                 num_heads=None,
                 mlp_ratio=3.0,
                 dropout=0.0,
                 attn_drop=0.0,
                 block_type='sr',
                 ode_steps_attn=2,
                 ode_steps_mlp=1,
                 ode_steps_fric=1,
                 use_friction=True,
                 ode_mode="strang",
                 friction_position='mid',
                 patch_size=4):
        super().__init__()
        print(f"assert len({len(embed_dim)}) == {len(depth)} == {len(sr_ratio)}")
        assert len(embed_dim) == len(depth) == len(sr_ratio)
        self.num_stages = len(embed_dim)
        if num_heads is None:
            num_heads = [max(1, d // 64) for d in embed_dim]

        self.patch_size = patch_size

        # Caution : patch size should be equivalent to the common value of it.
        # num_params = out_channels * (in_channels // groups) * kD * kH * kW
        # bytes = num_params * bytes_per_element
        self.patch_embed = ViT3DPatchEmbed(self.patch_size, in_channels, embed_dim[0])

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.has_cls_token = False

        for i in range(self.num_stages):
            dim = embed_dim[i]
            hd = num_heads[i]
            sr = sr_ratio[i]
            dth = depth[i]

            

            stage_blocks = nn.ModuleList()

            
            # stage_blocks.append(AttentionMLP(dim, 1 << i, sr))

            for _ in range(dth):
                if block_type == 'sr':
                    stage_blocks.append(SRtransformerBlock3D(dim, num_heads=hd, sr_ratio=sr, mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop))
                else:
                    attn_field = AttentionField3D(dim, num_heads=hd, dropout=dropout)
                    mlp_field = MLPField(dim, mlp_ratio=mlp_ratio, dropout=dropout)
                    fric_field = FrictionField(dim)
                    stage_blocks.append(
                        SplitODEBlock(
                            attn_field=attn_field,
                            mlp_field=mlp_field,
                            fric_field=fric_field,
                            steps_attn=ode_steps_attn,
                            steps_mlp=ode_steps_mlp,
                            steps_fric=ode_steps_fric,
                            use_friction=use_friction,
                            friction_position=friction_position,
                            ode_mode=ode_mode,
                        )
                    )
            self.stages.append(stage_blocks)

            if i < self.num_stages -1 :
                self.downs.append(PatchMerging3D(embed_dim[i], embed_dim[i + 1]))

    def _tokens_to_feat(self, tok, grid, has_cls=False):
        B, N, C = tok.shape
        D, L, W = grid

        if has_cls:
            if N < 2:
                raise ValueError(f'token count N = {N},if has cls is True, it requires more tokens')
            tok = tok[:, 1:, :]
            N = N - 1
        
        print("expected:", (grid[0]*grid[1]*grid[2]))
        assert N == D * L * W, (f'Token count mismatch: {N}, {grid} equivalent to {D*L*W}')
    
        feat = tok.transpose(1, 2).contiguous().view(B, C, D, L, W)
        return feat
    
    def forward(self, x):
        print("input:", x.shape)

        patch_out = self.patch_embed(x)
        feats = []

        if isinstance(patch_out, (tuple, list)) and len(patch_out) == 3:
            tok, grid, feat = patch_out
        elif isinstance(patch_out, (tuple, list)) and len(patch_out) == 2:
            tok, grid = patch_out
            feat = self._tokens_to_feat(tok, grid, self.has_cls_token)
        else:
            raise ValueError("ViTPatchEmbed must return (tok, grid, feat) or (tok, grid)")

        print("after patch_embed:", feat.shape, "grid:", grid)

        for i, stage_blocks in enumerate(self.stages):
            if i > 0:
                # print(f"before downs[{i - 1}]:", feat.shape)
                down = self.downs[i - 1](feat)

                if isinstance(down, (tuple, list)) and len(down) == 3:
                    tok, grid, feat = down
                elif isinstance(down, (tuple, list)) and len(down) == 2:
                    tok, grid = down
                    feat = self._tokens_to_feat(tok, grid, self.has_cls_token)
                else:
                    raise ValueError("PatchMerging must return (tok, grid, feat) or (tok, grid)")

                print(f"after downs[{i - 1}]:", feat.shape, "grid:", grid)

            for blk in stage_blocks:
                if isinstance(blk, (AttentionField3D, MLPField, FrictionField)):
                    tok = blk(0.0, tok, blk)
                elif isinstance(blk, (SplitODEBlock, SRTransformerBlock3D)):
                    tok = blk(tok, blk)
                else:
                    tok = blk(tok, grid)

            feat = self._tokens_to_feat(tok, grid, self.has_cls_token)
            print(f"after stage[{i}]:", feat.shape, "grid:", grid)
            feats.append(feat)

        feat_last = feats[-1]
        print([f.shape for f in feats])
        return feat_last, feats