"""
solvers for ji scripts : jl for computing derivatives from 3D datasets
"""


import torch
import torch.nn as nn
from typing import Optional, Tuple


class RKSolver(nn.Module):
    def __init__(self, vf: nn.Module):
        super().__init__()
        self.vf = vf

    def forward(self, t: torch.Tensor, x: torch.Tensor, dt: float, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        k1 = self.vf(t, x, grid_shape)
        k2 = self.vf(t + .5 * dt, x + .5 * dt * k1, grid_shape)
        k3 = self.vf(t + .5 * dt, x + .5 * dt * k2, grid_shape)
        k_f = self.vf(t + dt, x + dt * k3, grid_shape)
        k = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k_f)
        print(f'k: {k}')

        return k

class ODEInterateRK(nn.Module):
    def __init__(self, vf: nn.Module):
        super().__init__()
        self.solver = RKSolver(vf)
        # self.solver = torch.jit.script(RKSolver(vf))

    def forward(self, x0: torch.Tensor, grid_shape: Tuple[int, int, int], t_0 = 0.0, t_1 = 1.0, steps = 10) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        x = x0
        dt = torch.tensor((t_1 - t_0) / float(steps), dtype=x.dtype, device=x.device)
        t = torch.tensor(t_0, dtype=x.dtype, device = x.device)

        for _ in range(steps):
            x = self.solver(t, x, dt, grid_shape)
            t = t + dt

        return x

def ode_integrate_rk(vf: nn.Module, x0: torch.Tensor, t_0=0.0, t_1=1.0, steps=10):
    solver = ODEInterateRK(vf)

    return solver(x0, grid_shape, t_0=0.0, t_1=1.0, steps=20)