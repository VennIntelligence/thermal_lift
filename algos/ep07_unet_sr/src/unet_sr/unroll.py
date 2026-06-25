"""K-step physics-constrained unrolled super-resolution solver.

This is the committed redesign direction (ACL-024 / research_log/network_upgrade_roadmap.md).
It generalizes the V10 "residual-over-drizzle" model (which ACL-024 notes is already a 1-step
unroll) to K steps with a HARD data-consistency (DC) step between proximal refinements:

    x_0 = drizzle warm-start
    for k in 0..K-1:
        x <- x - eta_k * A^T(A x - y)        # DC gradient step (exact A^T via autograd)
        x <- x + prox_k([x, cond])           # learned proximal refinement (UNet)
    return x_K

Why this beats the falsified loss-side anchoring (ACL-017/019): the DC step constrains the
*range* of the forward operator with a HARD update every iteration, so the UNet prox can only
fill the forward operator's null space — it structurally cannot overwrite the observations the
way a soft forward-model loss let it (the fidelity-cliff mechanism FM-1/FM-2).

The forward operator A and its exact transpose A^T are certified in forward_torch.py (Gate A).
A^T comes from autograd, so the whole unroll is end-to-end differentiable (double-backward
during training — set create_graph via .training).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward_torch import ScenePSF, data_consistency_grad
from .model import ThermalSRUNet


def _inv_softplus(y: float) -> float:
    """raw such that softplus(raw) == y  (for initializing positive step sizes)."""
    return math.log(math.expm1(max(float(y), 1e-6)))


class UnrolledSolver(nn.Module):
    """K-step learned proximal-gradient solver over the certified physical forward operator.

    Args:
        n_steps:       K, number of unroll iterations (3-8 typical).
        cond_channels: channels of the conditioning tensor fed to each prox (e.g. 8 for the
                       hybrid_drizzle2x obs: 5 fused↑2x + 3 drizzle@2x).
        base_channels: UNet width for the prox net(s).
        scale:         SR factor (2).
        share_weights: one prox net reused across steps (data-efficient, default) vs per-step.
        band_highpass_sigma_lr_px: >0 runs the DC term in the high-frequency band (rejects the
                       smooth per-frame drift; matches the SR band, per the data audit).
        huber_delta:   >0 uses a robust Huber DC term (hot/cold defects, stripe noise).
        eta_init:      initial DC step size (learnable, softplus-parameterized, per step).
    """

    def __init__(
        self,
        *,
        n_steps: int = 4,
        cond_channels: int = 8,
        base_channels: int = 64,
        scale: int = 2,
        share_weights: bool = True,
        band_highpass_sigma_lr_px: float = 5.0,
        huber_delta: float = 0.0,
        eta_init: float = 0.5,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_steps = int(n_steps)
        self.scale = int(scale)
        self.share_weights = bool(share_weights)
        self.cond_channels = int(cond_channels)
        self.band_highpass_sigma_lr_px = float(band_highpass_sigma_lr_px)
        self.huber_delta = float(huber_delta)

        in_ch = 1 + self.cond_channels  # [current estimate x] ++ conditioning
        if self.share_weights:
            self.prox = ThermalSRUNet(in_channels=in_ch, out_channels=1, base_channels=base_channels, scale=1)
        else:
            self.prox = nn.ModuleList(
                ThermalSRUNet(in_channels=in_ch, out_channels=1, base_channels=base_channels, scale=1)
                for _ in range(self.n_steps)
            )
        # learnable positive per-step DC step sizes
        self.eta_raw = nn.Parameter(torch.full((self.n_steps,), _inv_softplus(eta_init)))

    def _prox_net(self, k: int) -> ThermalSRUNet:
        return self.prox if self.share_weights else self.prox[k]

    def _prox(self, k: int, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Residual proximal refinement: x <- x + net([x, cond]) (V10-style, stable)."""
        delta = self._prox_net(k)(torch.cat([x, cond], dim=1))
        return x + delta

    def forward(
        self,
        x0: torch.Tensor,
        y_burst: torch.Tensor,
        shifts: torch.Tensor,
        psf: ScenePSF,
        cond: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        return_iters: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the K-step solver.

        Args:
            x0:      (B,1,H,W) warm-start HR estimate (the drizzle mean, hybrid ch5).
            y_burst: (B,M,h,w) observed LR burst (the SAVED burst; FIXED M frames).
            shifts:  (B,M,2) the SAVED per-frame shifts in LR px (must match the burst).
            psf:     per-sample ScenePSF (from scene metadata: shape/sigma/sigma_y/angle).
            cond:    (B,cond_channels,H,W) conditioning for the prox (e.g. the hybrid obs).
            frame_mask: optional (B,M,1,1) weights (0 for padded/invalid frames).
        Returns: x_K, or (x_K, [x_1..x_K]) if return_iters.
        """
        if x0.shape[1] != 1:
            raise ValueError(f"x0 must be (B,1,H,W); got {tuple(x0.shape)}")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(f"cond must have {self.cond_channels} channels; got {cond.shape[1]}")
        create_graph = self.training
        huber = self.huber_delta if self.huber_delta > 0 else None
        x = x0
        iters: list[torch.Tensor] = []
        for k in range(self.n_steps):
            eta = F.softplus(self.eta_raw[k])
            g, _ = data_consistency_grad(
                x, y_burst, shifts, psf, self.scale,
                frame_mask=frame_mask,
                band_highpass_sigma_lr_px=self.band_highpass_sigma_lr_px,
                huber_delta=huber,
                create_graph=create_graph,
            )
            x = x - eta * g            # DC gradient step (range constraint)
            x = self._prox(k, x, cond)  # learned proximal step (null-space prior)
            if return_iters:
                iters.append(x)
        return (x, iters) if return_iters else x

    @torch.no_grad()
    def dc_residual_rms(
        self, x: torch.Tensor, y_burst: torch.Tensor, shifts: torch.Tensor, psf: ScenePSF,
        *, band: bool = True,
    ) -> torch.Tensor:
        """Monitoring: RMS of (A x - y) (optionally in the highpass band). Used by Gate B to
        confirm the DC residual drives toward 0 on a clean overfit scene."""
        from .forward_torch import _highpass, forward_burst
        r = forward_burst(x, shifts, psf, self.scale) - y_burst
        if band and self.band_highpass_sigma_lr_px > 0:
            r = _highpass(r, self.band_highpass_sigma_lr_px)
        return torch.sqrt((r * r).mean())
