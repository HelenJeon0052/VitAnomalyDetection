import torch
import torch.nn as nn



class GatedSkip(nn.Module):
    def __init__(self, in_main, in_skip, out):
        super().__init__()
        self.proj_main = nn.Conv3d(in_main, out, kernel_size=1)
        self.proj_skip = nn.Conv3d(in_skip, out, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv3d(out*2, out, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, main, skip):
        main = self.proj_main(main)
        skip = self.proj_skip(skip)
        g = self.gate(torch.cat([main, skip], dim=1))
        g_f = main + g * skip

        return g_f

class LightMLPRefine(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        l = int(dim * mlp_ratio)
        self.pw1 = nn.Conv3d(dim, l, kernel_size=1)
        self.pw2 = nn.Conv3d(l, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GeLU()

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        x = self.drop(x)

        return x






