import torch
import torch.nn as nn


from typing import Tuple

from AnomalyDetectionVit.solvers.rk import ode_integrate_rk


class SplitODEBlock(nn.Module):
    """
    residual block
    Compose sub-flows:
     - Lie-Trotter: f_fric^h > f_attn^h > f_mlp^(h)
     - Strang : f_attn^(h/2) > f_mlp^h > f_attn^(h/2) + optional friction insertion
    """
    def __init__(self,
                 attn_field: nn.Module,
                 mlp_field: nn.Module,
                 fric_field: nn.Module,
                 mode: str = 'strang',
                 steps_attn: int = 2,
                 steps_mlp: int = 1,
                 steps_fric: int = 1,
                 t_0: float = 0.0,
                 t_1: float = 1.0,
                 use_friction: bool = True,
                 friction_position: str = 'mid'
    ):
        super().__init__()
        assert mode in ['lie', 'strang']
        assert friction_position in ['pre', 'mid', 'post']

        self.attn = attn_field
        self.mlp = mlp_field
        self.fric = fric_field
        self.mode = mode

        self.steps_attn = steps_attn
        self.steps_mlp = steps_mlp
        self.steps_fric = steps_fric

        self.t_0, self.t_1 = t_0, t_1
        self.use_friction = use_friction and (fric_field is not None)
        self.friction_position = friction_position

    def _flow(self, field: nn.Module, x: torch.Tensor, grid_shape: Tuple[int, int, int], t_0, t_1, steps) -> torch.Tensor:
        return  ode_integrate_rk(field, x, grid_shape, t_0 = t_0, t_1 = t_1, steps = steps)

    """def _fric(self, x, grid_shape):
        if not self.use_friction:
            return x
        return self._flow(self.fric, x, grid_shape, self.steps_fric, t_0=0.0, t_1=1.0)"""

    def forward(self, x, grid_shape):
        t_0, t_1 = self.t_0, self.t_1
        mid = 0.5 * (t_0 + t_1)

        def apply_fric(z, grid_shape):
            if self.use_friction:
                return self._flow(self.fric, z, grid_shape, t_0, t_1, self.steps_fric)
            return z

        if self.mode == 'lie':
            # Lie - Trotter : Fric > attn > Fric > Fric
            if self.friction_position == 'pre':
                x = apply_fric(x, grid_shape)

            x = self._flow(self.attn, x, grid_shape, t_0, t_1, self.steps_attn)

            if self.friction_position == 'mid':
                x = apply_fric(x, grid_shape)
            x = self._flow(self.mlp, x, grid_shape, t_0, t_1, self.steps_mlp)

            if self.friction_position == 'post':
                x = apply_fric(x, grid_shape)

            return x

        # Strang :  attn(h/2) > (optional fric) > mlp(h) > attn(h/2)
        if self.mode == "strang":
            x = self._flow(self.attn, x, grid_shape, t_0, mid, max(1, self.steps_attn // 2))

            if self.friction_position == 'mid':
               x = apply_fric(x, grid_shape)

            x = self._flow(self.mlp, x, grid_shape, t_0, t_1, self.steps_mlp)
            x = self._flow(self.attn, x, grid_shape, mid, t_1, max(1, self.steps_attn//2))

            if self.friction_position == 'post':
              x = apply_fric(x, grid_shape)
            elif self.friction_position == 'pre':
              x = apply_fric(x, grid_shape)

            return x